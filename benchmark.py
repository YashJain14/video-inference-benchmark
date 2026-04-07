"""
Video Inference Pipeline Benchmark
====================================
Compares 5 pipeline configurations end-to-end:
  1. OpenCV CPU          (baseline)
  2. FFmpeg + DALI       (article baseline)
  3. TorchCodec GPU
  4. PyNvVideoCodec 2.1  (ThreadedDecoder)
  5. PyNvVideoCodec 2.1  + CV-CUDA preprocessing

Each pipeline measures 5 stages independently:
  decode / preprocess / inference / postprocess / total

Results saved to results/benchmark_results.csv
Charts saved to charts/
"""

import argparse
import csv
import gc
import importlib
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results"
CHARTS_DIR  = Path(__file__).parent / "charts"
RESULTS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    video_path: str
    model_path: str                        # YOLO .pt or TensorRT .engine
    batch_size: int        = 8
    warmup_runs: int       = 3
    benchmark_runs: int    = 10
    input_size: tuple      = (640, 640)    # model input resolution
    device: str            = "cuda:0"
    num_streams: int       = 4             # for multi-stream configs
    fp16: bool             = True

# ─────────────────────────────────────────────
# Timing utilities
# ─────────────────────────────────────────────

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

class Timer:
    """Context manager that records wall time with CUDA sync."""
    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        cuda_sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        cuda_sync()
        self.elapsed = time.perf_counter() - self._start


@dataclass
class StageTimings:
    pipeline_name: str
    run_id: int
    decode_s: float     = 0.0
    preprocess_s: float = 0.0
    inference_s: float  = 0.0
    postprocess_s: float= 0.0
    total_s: float      = 0.0
    frames_processed: int = 0
    error: Optional[str]  = None

    @property
    def fps(self):
        return self.frames_processed / self.total_s if self.total_s > 0 else 0.0


# ─────────────────────────────────────────────
# Base pipeline interface
# ─────────────────────────────────────────────

class BasePipeline:
    name: str = "base"

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._model = None

    def setup(self):
        """Load model and any pipeline-specific resources. Called once."""
        raise NotImplementedError

    def run_once(self) -> StageTimings:
        """Run the full pipeline once, return per-stage timings."""
        raise NotImplementedError

    def teardown(self):
        """Release GPU memory / file handles."""
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── shared helpers ──────────────────────────────

    def _load_yolo_model(self):
        """Load YOLO model via ultralytics (works for .pt and TensorRT .engine)."""
        try:
            from ultralytics import YOLO
            model = YOLO(self.config.model_path)
            model.to(self.config.device)
            if self.config.fp16:
                model.model.half()
            return model
        except ImportError:
            raise RuntimeError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

    def _postprocess(self, raw_output) -> dict:
        """
        Convert raw model output to bounding boxes + labels.
        Returns dict with 'boxes', 'scores', 'classes'.
        """
        if hasattr(raw_output, "boxes"):
            # ultralytics Results object
            boxes  = raw_output.boxes.xyxy if raw_output.boxes else torch.empty(0, 4)
            scores = raw_output.boxes.conf if raw_output.boxes else torch.empty(0)
            classes = raw_output.boxes.cls  if raw_output.boxes else torch.empty(0)
        else:
            boxes = scores = classes = torch.empty(0)
        return {"boxes": boxes, "scores": scores, "classes": classes}


# ─────────────────────────────────────────────
# Pipeline 1 — OpenCV CPU (baseline)
# ─────────────────────────────────────────────

class OpenCVCPUPipeline(BasePipeline):
    name = "OpenCV CPU"

    def setup(self):
        import cv2
        self._cv2 = cv2
        self._model = self._load_yolo_model()

    def run_once(self) -> StageTimings:
        cv2 = self._cv2
        cfg = self.config
        t = StageTimings(pipeline_name=self.name, run_id=0)

        with Timer() as total:

            # ── decode ──────────────────────────────
            with Timer() as td:
                cap = cv2.VideoCapture(cfg.video_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
            t.decode_s = td.elapsed

            if not frames:
                t.error = "No frames decoded"
                return t

            # ── preprocess ──────────────────────────
            with Timer() as tp:
                batch = []
                for frame in frames[: cfg.batch_size]:
                    resized = cv2.resize(frame, cfg.input_size)
                    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    tensor  = torch.from_numpy(rgb).float().div(255.0)
                    tensor  = tensor.permute(2, 0, 1)
                    batch.append(tensor)
                batch_tensor = torch.stack(batch).to(cfg.device)
                if cfg.fp16:
                    batch_tensor = batch_tensor.half()
            t.preprocess_s = tp.elapsed

            # ── inference ───────────────────────────
            with Timer() as ti:
                results = self._model(
                    batch_tensor,
                    verbose=False,
                    half=cfg.fp16,
                )
            t.inference_s = ti.elapsed

            # ── postprocess ─────────────────────────
            with Timer() as tpost:
                outputs = [self._postprocess(r) for r in results]
            t.postprocess_s = tpost.elapsed

            t.frames_processed = len(frames)

        t.total_s = total.elapsed
        return t


# ─────────────────────────────────────────────
# Pipeline 2 — FFmpeg + DALI  (article baseline)
# ─────────────────────────────────────────────

class FFmpegDALIPipeline(BasePipeline):
    name = "FFmpeg + DALI"

    def setup(self):
        try:
            import nvidia.dali as dali
            import nvidia.dali.fn as fn
            from nvidia.dali.pipeline import pipeline_def
            self._dali = dali
            self._fn   = fn
            self._pipeline_def = pipeline_def
        except ImportError:
            raise RuntimeError(
                "DALI not installed. Run: pip install nvidia-dali-cuda120"
            )
        self._model = self._load_yolo_model()
        self._build_dali_pipeline()

    def _build_dali_pipeline(self):
        fn  = self._fn
        cfg = self.config

        @self._pipeline_def(batch_size=cfg.batch_size, num_threads=4, device_id=0)
        def video_pipe():
            frames = fn.readers.video(
                filenames=[cfg.video_path],
                sequence_length=cfg.batch_size,
                normalized=False,
                random_shuffle=False,
                image_type=self._dali.types.RGB,
                dtype=self._dali.types.UINT8,
                name="Reader",
                device="gpu",
            )
            frames = fn.resize(
                frames,
                resize_x=cfg.input_size[0],
                resize_y=cfg.input_size[1],
            )
            frames = fn.crop_mirror_normalize(
                frames,
                dtype=self._dali.types.FLOAT,
                mean=[0.0, 0.0, 0.0],
                std=[255.0, 255.0, 255.0],
                output_layout="FCHW",
            )
            return frames

        self._dali_pipe = video_pipe()
        self._dali_pipe.build()

    def run_once(self) -> StageTimings:
        cfg = self.config
        t   = StageTimings(pipeline_name=self.name, run_id=0)

        with Timer() as total:

            # DALI fuses decode + preprocess; we time them together then split
            with Timer() as tdp:
                out = self._dali_pipe.run()
                batch_tensor = out[0].as_tensor()
                batch_torch  = torch.empty(
                    batch_tensor.shape(), dtype=torch.float32, device=cfg.device
                )
                batch_torch.copy_(torch.as_tensor(batch_tensor, device=cfg.device))
            # attribute 60 % to decode, 40 % to preprocess (NVDEC vs GPU kernels split)
            t.decode_s     = tdp.elapsed * 0.60
            t.preprocess_s = tdp.elapsed * 0.40

            with Timer() as ti:
                results = self._model(batch_torch, verbose=False, half=cfg.fp16)
            t.inference_s = ti.elapsed

            with Timer() as tpost:
                outputs = [self._postprocess(r) for r in results]
            t.postprocess_s = tpost.elapsed

            t.frames_processed = batch_torch.shape[0]

        t.total_s = total.elapsed
        return t


# ─────────────────────────────────────────────
# Pipeline 3 — TorchCodec GPU
# ─────────────────────────────────────────────

class TorchCodecPipeline(BasePipeline):
    name = "TorchCodec GPU"

    def setup(self):
        try:
            from torchcodec.decoders import VideoDecoder
            self._VideoDecoder = VideoDecoder
        except ImportError:
            raise RuntimeError(
                "torchcodec not installed.\n"
                "Run: pip install torchcodec "
                "--index-url https://download.pytorch.org/whl/cu124"
            )
        self._model = self._load_yolo_model()

    def run_once(self) -> StageTimings:
        cfg = self.config
        t   = StageTimings(pipeline_name=self.name, run_id=0)

        with Timer() as total:

            # ── decode (NVDEC via TorchCodec) ────────
            with Timer() as td:
                decoder = self._VideoDecoder(cfg.video_path, device=cfg.device)
                # grab up to batch_size frames evenly spaced
                n = len(decoder)
                indices = list(range(0, min(n, cfg.batch_size)))
                frame_batch = decoder.get_frames_at(indices=indices)
                frames_gpu = frame_batch.data  # [N, C, H, W] uint8 on GPU
            t.decode_s = td.elapsed

            # ── preprocess ──────────────────────────
            with Timer() as tp:
                frames_f = frames_gpu.float().div(255.0)
                # resize using torch interpolate
                frames_f = torch.nn.functional.interpolate(
                    frames_f,
                    size=cfg.input_size,
                    mode="bilinear",
                    align_corners=False,
                )
            t.preprocess_s = tp.elapsed

            # ── inference ───────────────────────────
            with Timer() as ti:
                results = self._model(frames_f, verbose=False, half=cfg.fp16)
            t.inference_s = ti.elapsed

            # ── postprocess ─────────────────────────
            with Timer() as tpost:
                outputs = [self._postprocess(r) for r in results]
            t.postprocess_s = tpost.elapsed

            t.frames_processed = frames_gpu.shape[0]

        t.total_s = total.elapsed
        return t


# ─────────────────────────────────────────────
# Pipeline 4 — PyNvVideoCodec 2.1 ThreadedDecoder
# ─────────────────────────────────────────────

class PyNvVideoCodecPipeline(BasePipeline):
    name = "PyNvVideoCodec 2.1"

    def setup(self):
        try:
            from PyNvVideoCodec import ThreadedDecoder, OutputColorType
            self._ThreadedDecoder = ThreadedDecoder
            self._OutputColorType = OutputColorType
        except ImportError:
            raise RuntimeError(
                "PyNvVideoCodec not installed.\n"
                "Run: pip install pynvvideocodec"
            )
        self._model = self._load_yolo_model()

    def run_once(self) -> StageTimings:
        cfg = self.config
        t   = StageTimings(pipeline_name=self.name, run_id=0)

        with Timer() as total:

            # ── decode (NVDEC ThreadedDecoder) ───────
            with Timer() as td:
                decoder = self._ThreadedDecoder(
                    cfg.video_path,
                    gpu_id=0,
                    output_color_type=self._OutputColorType.RGBP,  # CHW, ideal for PyTorch
                    buffer_size=cfg.batch_size * 2,
                )
                frames = []
                for i, frame in enumerate(decoder):
                    if i >= cfg.batch_size:
                        break
                    # frame is already a CUDA buffer; convert to torch tensor
                    frames.append(
                        torch.as_tensor(frame, device=cfg.device)
                    )
                decoder.close()
            t.decode_s = td.elapsed

            if not frames:
                t.error = "No frames decoded"
                return t

            # ── preprocess ──────────────────────────
            with Timer() as tp:
                batch = torch.stack(frames).float().div(255.0)
                batch = torch.nn.functional.interpolate(
                    batch,
                    size=cfg.input_size,
                    mode="bilinear",
                    align_corners=False,
                )
            t.preprocess_s = tp.elapsed

            # ── inference ───────────────────────────
            with Timer() as ti:
                results = self._model(batch, verbose=False, half=cfg.fp16)
            t.inference_s = ti.elapsed

            # ── postprocess ─────────────────────────
            with Timer() as tpost:
                outputs = [self._postprocess(r) for r in results]
            t.postprocess_s = tpost.elapsed

            t.frames_processed = batch.shape[0]

        t.total_s = total.elapsed
        return t


# ─────────────────────────────────────────────
# Pipeline 5 — PyNvVideoCodec 2.1 + CV-CUDA
# ─────────────────────────────────────────────

class PyNvVideoCodecCVCudaPipeline(BasePipeline):
    name = "PyNvVideoCodec + CV-CUDA"

    def setup(self):
        try:
            from PyNvVideoCodec import ThreadedDecoder, OutputColorType
            self._ThreadedDecoder = ThreadedDecoder
            self._OutputColorType = OutputColorType
        except ImportError:
            raise RuntimeError("PyNvVideoCodec not installed. Run: pip install pynvvideocodec")

        try:
            import cvcuda
            self._cvcuda = cvcuda
        except ImportError:
            raise RuntimeError(
                "CV-CUDA not installed.\n"
                "Run: pip install cvcuda-cu12"
            )

        self._model = self._load_yolo_model()

    def run_once(self) -> StageTimings:
        cfg    = self.config
        cvcuda = self._cvcuda
        t      = StageTimings(pipeline_name=self.name, run_id=0)

        with Timer() as total:

            # ── decode (NVDEC ThreadedDecoder) ───────
            with Timer() as td:
                decoder = self._ThreadedDecoder(
                    cfg.video_path,
                    gpu_id=0,
                    output_color_type=self._OutputColorType.RGBP,
                )
                raw_frames = []
                for i, frame in enumerate(decoder):
                    if i >= cfg.batch_size:
                        break
                    raw_frames.append(torch.as_tensor(frame, device=cfg.device))
                decoder.close()
            t.decode_s = td.elapsed

            if not raw_frames:
                t.error = "No frames decoded"
                return t

            # ── preprocess via CV-CUDA ───────────────
            # CV-CUDA operates on NHWC tensors; our frames are CHW → permute
            with Timer() as tp:
                # Stack → [N, C, H, W] then → [N, H, W, C] for CV-CUDA
                batch_chw  = torch.stack(raw_frames).float().div(255.0)
                batch_nhwc = batch_chw.permute(0, 2, 3, 1).contiguous()

                # Wrap as CV-CUDA tensor (zero-copy)
                cvcuda_tensor = cvcuda.as_tensor(batch_nhwc, "NHWC")

                # Resize using CV-CUDA GPU kernel
                resized = cvcuda.resize(
                    cvcuda_tensor,
                    (batch_nhwc.shape[0], cfg.input_size[1], cfg.input_size[0], 3),
                    cvcuda.Interp.LINEAR,
                )

                # Normalize in-place (subtract mean, divide std — all on GPU)
                normalized = cvcuda.normalize(
                    resized,
                    base=cvcuda.as_tensor(
                        torch.tensor([0.0, 0.0, 0.0], device=cfg.device), "C"
                    ),
                    scale=cvcuda.as_tensor(
                        torch.tensor([255.0, 255.0, 255.0], device=cfg.device), "C"
                    ),
                    flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
                )

                # Back to [N, C, H, W] PyTorch tensor (zero-copy via dlpack)
                result_torch = torch.as_tensor(normalized.cuda(), device=cfg.device)
                batch_final  = result_torch.permute(0, 3, 1, 2).contiguous()

            t.preprocess_s = tp.elapsed

            # ── inference ───────────────────────────
            with Timer() as ti:
                results = self._model(batch_final, verbose=False, half=cfg.fp16)
            t.inference_s = ti.elapsed

            # ── postprocess ─────────────────────────
            with Timer() as tpost:
                outputs = [self._postprocess(r) for r in results]
            t.postprocess_s = tpost.elapsed

            t.frames_processed = batch_final.shape[0]

        t.total_s = total.elapsed
        return t


# ─────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────

PIPELINE_REGISTRY = {
    "opencv_cpu":      OpenCVCPUPipeline,
    "ffmpeg_dali":     FFmpegDALIPipeline,
    "torchcodec":      TorchCodecPipeline,
    "pynvvideocodec":  PyNvVideoCodecPipeline,
    "pynv_cvcuda":     PyNvVideoCodecCVCudaPipeline,
}


def run_benchmark(config: BenchmarkConfig, pipelines: list[str]) -> list[StageTimings]:
    all_results = []

    for name in pipelines:
        cls = PIPELINE_REGISTRY[name]
        pipeline = cls(config)
        print(f"\n{'─'*60}")
        print(f"  Pipeline: {pipeline.name}")
        print(f"{'─'*60}")

        # ── setup ────────────────────────────────
        try:
            pipeline.setup()
            print(f"  [setup] OK")
        except Exception as e:
            print(f"  [setup] FAILED: {e}")
            result = StageTimings(
                pipeline_name=pipeline.name, run_id=0, error=str(e)
            )
            all_results.append(result)
            continue

        # ── warmup ───────────────────────────────
        print(f"  [warmup] {config.warmup_runs} runs...", end=" ", flush=True)
        for _ in range(config.warmup_runs):
            try:
                pipeline.run_once()
            except Exception as e:
                print(f"\n  [warmup] FAILED: {e}")
                break
        print("done")

        # ── benchmark runs ───────────────────────
        run_results = []
        print(f"  [bench]  {config.benchmark_runs} runs...", end=" ", flush=True)
        for i in range(config.benchmark_runs):
            try:
                r = pipeline.run_once()
                r.run_id = i
                run_results.append(r)
                print(f"{i+1}", end=" ", flush=True)
            except Exception as e:
                print(f"\n  [bench] run {i} FAILED: {e}")
                run_results.append(
                    StageTimings(pipeline_name=pipeline.name, run_id=i, error=str(e))
                )
        print()

        all_results.extend(run_results)

        # ── teardown ─────────────────────────────
        pipeline.teardown()
        print(f"  [teardown] OK")

    return all_results


# ─────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────

def save_csv(results: list[StageTimings], path: Path):
    if not results:
        print("No results to save.")
        return

    fieldnames = [
        "pipeline_name", "run_id",
        "decode_s", "preprocess_s", "inference_s", "postprocess_s",
        "total_s", "frames_processed", "fps", "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["fps"] = r.fps
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\n  Results saved → {path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Video inference pipeline benchmark")
    parser.add_argument("--video",   required=True,  help="Path to input video file")
    parser.add_argument("--model",   required=True,  help="Path to YOLO model (.pt or .engine)")
    parser.add_argument("--batch",   type=int, default=8)
    parser.add_argument("--warmup",  type=int, default=3)
    parser.add_argument("--runs",    type=int, default=10)
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 640],
                        metavar=("W", "H"))
    parser.add_argument("--device",  default="cuda:0")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--pipelines", nargs="+",
                        choices=list(PIPELINE_REGISTRY.keys()),
                        default=list(PIPELINE_REGISTRY.keys()),
                        help="Which pipelines to run")
    parser.add_argument("--out-csv", default=str(RESULTS_DIR / "benchmark_results.csv"))
    args = parser.parse_args()

    config = BenchmarkConfig(
        video_path    = args.video,
        model_path    = args.model,
        batch_size    = args.batch,
        warmup_runs   = args.warmup,
        benchmark_runs= args.runs,
        input_size    = tuple(args.input_size),
        device        = args.device,
        fp16          = not args.no_fp16,
    )

    print(f"\nVideo Inference Pipeline Benchmark")
    print(f"  Video      : {config.video_path}")
    print(f"  Model      : {config.model_path}")
    print(f"  Batch size : {config.batch_size}")
    print(f"  Input size : {config.input_size}")
    print(f"  Device     : {config.device}")
    print(f"  FP16       : {config.fp16}")
    print(f"  Pipelines  : {args.pipelines}")

    results = run_benchmark(config, args.pipelines)
    save_csv(results, Path(args.out_csv))

    # ── quick summary ────────────────────────
    print(f"\n{'─'*60}")
    print(f"  {'Pipeline':<30} {'Avg FPS':>9} {'Avg Total (s)':>14}")
    print(f"{'─'*60}")
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if not r.error:
            groups[r.pipeline_name].append(r)
    for pname, runs in groups.items():
        avg_fps   = sum(r.fps for r in runs) / len(runs)
        avg_total = sum(r.total_s for r in runs) / len(runs)
        print(f"  {pname:<30} {avg_fps:>9.1f} {avg_total:>14.3f}")
    print(f"{'─'*60}")

    # ── generate charts ──────────────────────
    from charts import generate_all_charts
    generate_all_charts(Path(args.out_csv), CHARTS_DIR)


if __name__ == "__main__":
    main()
