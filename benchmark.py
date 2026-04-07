"""
Video Inference Pipeline Benchmark
====================================
Compares 5 pipeline configurations end-to-end:
  1. OpenCV CPU                (baseline)
  2. FFmpeg + DALI             (article baseline)
  3. TorchCodec GPU            (skipped if libnppicc missing)
  4. PyNvVideoCodec 2.1        (ThreadedDecoder)
  5. PyNvVideoCodec 2.1        + CV-CUDA preprocessing
"""

import argparse
import csv
import gc
import time
import warnings
from dataclasses import dataclass, asdict
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
    model_path: str
    batch_size: int     = 8
    warmup_runs: int    = 3
    benchmark_runs: int = 10
    input_size: tuple   = (640, 640)
    device: str         = "cuda:0"
    fp16: bool          = True

# ─────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

class Timer:
    def __init__(self): self.elapsed = 0.0
    def __enter__(self):
        cuda_sync(); self._start = time.perf_counter(); return self
    def __exit__(self, *_):
        cuda_sync(); self.elapsed = time.perf_counter() - self._start

@dataclass
class StageTimings:
    pipeline_name: str
    run_id: int
    decode_s: float      = 0.0
    preprocess_s: float  = 0.0
    inference_s: float   = 0.0
    postprocess_s: float = 0.0
    total_s: float       = 0.0
    frames_processed: int = 0
    error: Optional[str] = None

    @property
    def fps(self):
        return self.frames_processed / self.total_s if self.total_s > 0 else 0.0

# ─────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────

class BasePipeline:
    name: str = "base"
    def __init__(self, config): self.config = config; self._model = None
    def setup(self): raise NotImplementedError
    def run_once(self): raise NotImplementedError
    def teardown(self):
        self._model = None; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _load_yolo_model(self):
        # Do NOT call .half() — ultralytics handles dtype via half= on __call__
        try:
            from ultralytics import YOLO
            m = YOLO(self.config.model_path)
            m.to(self.config.device)
            return m
        except ImportError:
            raise RuntimeError("pip install ultralytics")

    def _infer(self, x):
        return self._model(x, verbose=False, half=self.config.fp16)

    def _postprocess(self, r):
        if hasattr(r, "boxes") and r.boxes is not None:
            return {"boxes": r.boxes.xyxy, "scores": r.boxes.conf, "classes": r.boxes.cls}
        return {"boxes": torch.empty(0), "scores": torch.empty(0), "classes": torch.empty(0)}

# ─────────────────────────────────────────────
# Pipeline 1 — OpenCV CPU
# ─────────────────────────────────────────────

class OpenCVCPUPipeline(BasePipeline):
    name = "OpenCV CPU"
    def setup(self):
        import cv2; self._cv2 = cv2
        self._model = self._load_yolo_model()

    def run_once(self):
        cv2, cfg = self._cv2, self.config
        t = StageTimings(pipeline_name=self.name, run_id=0)
        with Timer() as total:
            with Timer() as td:
                cap = cv2.VideoCapture(cfg.video_path)
                frames = []
                while True:
                    ret, f = cap.read()
                    if not ret: break
                    frames.append(f)
                cap.release()
            t.decode_s = td.elapsed
            if not frames: t.error = "No frames"; return t

            with Timer() as tp:
                batch = []
                for f in frames[:cfg.batch_size]:
                    r = cv2.resize(f, cfg.input_size)
                    r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
                    batch.append(torch.from_numpy(r).float().div(255.0).permute(2,0,1))
                batch_tensor = torch.stack(batch)
            t.preprocess_s = tp.elapsed

            with Timer() as ti: results = self._infer(batch_tensor)
            t.inference_s = ti.elapsed
            with Timer() as tp2: [self._postprocess(r) for r in results]
            t.postprocess_s = tp2.elapsed
            t.frames_processed = len(frames)
        t.total_s = total.elapsed
        return t

# ─────────────────────────────────────────────
# Pipeline 2 — FFmpeg + DALI
# ─────────────────────────────────────────────

class FFmpegDALIPipeline(BasePipeline):
    name = "FFmpeg + DALI"
    def setup(self):
        try:
            import nvidia.dali as dali, nvidia.dali.fn as fn
            from nvidia.dali.pipeline import pipeline_def
            self._dali = dali; self._fn = fn; self._pd = pipeline_def
        except ImportError:
            raise RuntimeError("pip install nvidia-dali-cuda120")
        self._model = self._load_yolo_model()
        self._build()

    def _build(self):
        fn, dali, cfg = self._fn, self._dali, self.config
        B = cfg.batch_size

        @self._pd(batch_size=B, num_threads=4, device_id=0)
        def pipe():
            # batch_size=B, sequence_length=1: DALI video reader outputs
            # each sample as [H, W, C] uint8 directly — NO sequence dim to squeeze.
            # The batch dimension is handled by DALI internally as a TensorList.
            frames = fn.readers.video(
                filenames=[cfg.video_path],
                sequence_length=1,
                normalized=False,
                random_shuffle=False,
                image_type=dali.types.RGB,
                dtype=dali.types.UINT8,
                name="Reader",
                device="gpu",
            )
            # Video reader with sequence_length=1 outputs layout FHWC per sample.
            # fn.resize keeps that layout: FHWC
            frames = fn.resize(frames,
                               resize_x=cfg.input_size[0],
                               resize_y=cfg.input_size[1])
            # output_layout must be a permutation of FHWC -> use FCHW
            # Result per sample: [1, C, H, W]; batch TensorList: B x [1, C, H, W]
            frames = fn.crop_mirror_normalize(
                frames,
                dtype=dali.types.FLOAT,
                mean=[0.0, 0.0, 0.0],
                std=[255.0, 255.0, 255.0],
                output_layout="FCHW",
            )
            return frames

        self._pipe = pipe(); self._pipe.build()

    def run_once(self):
        cfg = self.config
        t   = StageTimings(pipeline_name=self.name, run_id=0)
        with Timer() as total:
            with Timer() as tdp:
                out = self._pipe.run()
                dt  = out[0].as_tensor()
                bt  = torch.empty(dt.shape(), dtype=torch.float32, device=cfg.device)
                bt.copy_(torch.as_tensor(dt, device=cfg.device))
                # dt shape is [B, 1, C, H, W] (FCHW with F=1) — squeeze F dim
                if bt.ndim == 5:
                    bt = bt.squeeze(1)   # -> [B, C, H, W]
            # DALI fuses decode+preprocess; attribute 60/40
            t.decode_s     = tdp.elapsed * 0.60
            t.preprocess_s = tdp.elapsed * 0.40

            with Timer() as ti: results = self._infer(bt)
            t.inference_s = ti.elapsed
            with Timer() as tp: [self._postprocess(r) for r in results]
            t.postprocess_s = tp.elapsed
            t.frames_processed = bt.shape[0]
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
        except Exception as e:
            raise RuntimeError(
                f"TorchCodec failed: {e}\n"
                "libnppicc.so.12 not found on this cluster.\n"
                "Fix: conda install -c nvidia cuda-nppc"
            )
        self._model = self._load_yolo_model()

    def run_once(self):
        cfg = self.config
        t   = StageTimings(pipeline_name=self.name, run_id=0)
        with Timer() as total:
            with Timer() as td:
                dec  = self._VideoDecoder(cfg.video_path, device=cfg.device)
                idx  = list(range(0, min(len(dec), cfg.batch_size)))
                fb   = dec.get_frames_at(indices=idx)
                fgpu = fb.data
            t.decode_s = td.elapsed

            with Timer() as tp:
                f = fgpu.float().div(255.0)
                f = torch.nn.functional.interpolate(
                    f, size=cfg.input_size, mode="bilinear", align_corners=False)
            t.preprocess_s = tp.elapsed

            with Timer() as ti: results = self._infer(f)
            t.inference_s = ti.elapsed
            with Timer() as tp2: [self._postprocess(r) for r in results]
            t.postprocess_s = tp2.elapsed
            t.frames_processed = fgpu.shape[0]
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
            self._TD = ThreadedDecoder; self._OCT = OutputColorType
        except ImportError:
            raise RuntimeError("pip install pynvvideocodec")
        self._model = self._load_yolo_model()

    def _decode(self, n):
        """
        FIX: Use OutputColorType.RGB (interleaved HWC) instead of RGBP (planar).
        RGBP produces a non-contiguous planar buffer that torch.as_tensor()
        can't map cleanly -> CUDA invalid argument error.
        RGB gives a contiguous HWC uint8 tensor we can permute to CHW in PyTorch.
        """
        cfg = self.config
        dec = self._TD(
            cfg.video_path,
            n * 2,
            gpu_id=0,
            output_color_type=self._OCT.RGB,   # interleaved HWC — safe for as_tensor
        )
        batch = dec.get_batch_frames(n)
        dec.end()
        # batch is a list of DecodedFrame objects or raw tensors
        frames = []
        for f in batch:
            t = torch.as_tensor(f, device=cfg.device)
            if t.ndim == 2:
                # NV12/YUV planar — shouldn't happen with RGB, but guard anyway
                raise RuntimeError(f"Unexpected frame shape {t.shape}, expected HWC")
            # Expected: [H, W, C] uint8
            frames.append(t.clone())
        return frames

    def run_once(self):
        cfg = self.config
        t   = StageTimings(pipeline_name=self.name, run_id=0)
        with Timer() as total:
            with Timer() as td:
                frames = self._decode(cfg.batch_size)
            t.decode_s = td.elapsed
            if not frames: t.error = "No frames decoded"; return t

            with Timer() as tp:
                # Stack [H,W,C] frames -> [N,H,W,C] then permute -> [N,C,H,W]
                batch = torch.stack(frames).float().div(255.0).permute(0,3,1,2).contiguous()
                batch = torch.nn.functional.interpolate(
                    batch, size=cfg.input_size, mode="bilinear", align_corners=False)
            t.preprocess_s = tp.elapsed

            with Timer() as ti: results = self._infer(batch)
            t.inference_s = ti.elapsed
            with Timer() as tp2: [self._postprocess(r) for r in results]
            t.postprocess_s = tp2.elapsed
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
            self._TD = ThreadedDecoder; self._OCT = OutputColorType
        except ImportError:
            raise RuntimeError("pip install pynvvideocodec")
        try:
            import cvcuda; self._cvcuda = cvcuda
        except ImportError:
            raise RuntimeError("pip install cvcuda-cu12")
        self._model = self._load_yolo_model()

    def _decode(self, n):
        """Same RGB fix as Pipeline 4."""
        cfg = self.config
        dec = self._TD(cfg.video_path, n*2, gpu_id=0,
                       output_color_type=self._OCT.RGB)
        batch = dec.get_batch_frames(n)
        dec.end()
        frames = []
        for f in batch:
            frames.append(torch.as_tensor(f, device=cfg.device).clone())
        return frames

    def run_once(self):
        cfg, cvcuda = self.config, self._cvcuda
        t = StageTimings(pipeline_name=self.name, run_id=0)
        with Timer() as total:
            with Timer() as td: frames = self._decode(cfg.batch_size)
            t.decode_s = td.elapsed
            if not frames: t.error = "No frames"; return t

            with Timer() as tp:
                # frames are [H,W,C] uint8 — stack to [N,H,W,C] for CV-CUDA
                nhwc = torch.stack(frames).float().div(255.0).contiguous()

                cvcuda_in = cvcuda.as_tensor(nhwc, "NHWC")

                # Resize
                rsz = cvcuda.resize(
                    cvcuda_in,
                    (nhwc.shape[0], cfg.input_size[1], cfg.input_size[0], 3),
                    cvcuda.Interp.LINEAR,
                )

                # FIX: normalize with plain float scale/offset, not a "C"-layout tensor.
                # cvcuda.normalize base/scale as scalar tensors with layout "1" (scalar).
                scale = 1.0 / 255.0
                nrm = cvcuda.convertto(rsz, cvcuda.Type.F32, scale=scale, offset=0.0)

                # Back to [N, C, H, W] PyTorch
                result = torch.as_tensor(nrm.cuda(), device=cfg.device)
                final  = result.permute(0,3,1,2).contiguous()
            t.preprocess_s = tp.elapsed

            with Timer() as ti: results = self._infer(final)
            t.inference_s = ti.elapsed
            with Timer() as tp2: [self._postprocess(r) for r in results]
            t.postprocess_s = tp2.elapsed
            t.frames_processed = final.shape[0]
        t.total_s = total.elapsed
        return t

# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

PIPELINE_REGISTRY = {
    "opencv_cpu":     OpenCVCPUPipeline,
    "ffmpeg_dali":    FFmpegDALIPipeline,
    "torchcodec":     TorchCodecPipeline,
    "pynvvideocodec": PyNvVideoCodecPipeline,
    "pynv_cvcuda":    PyNvVideoCodecCVCudaPipeline,
}

def run_benchmark(config, pipelines):
    all_results = []
    for name in pipelines:
        pipeline = PIPELINE_REGISTRY[name](config)
        print(f"\n{'─'*60}\n  Pipeline: {pipeline.name}\n{'─'*60}")
        try:
            pipeline.setup(); print("  [setup] OK")
        except Exception as e:
            print(f"  [setup] FAILED: {e}")
            all_results.append(StageTimings(
                pipeline_name=pipeline.name, run_id=0, error=str(e)))
            continue

        print(f"  [warmup] {config.warmup_runs} runs...", end=" ", flush=True)
        warmup_ok = True
        for _ in range(config.warmup_runs):
            try: pipeline.run_once()
            except Exception as e:
                print(f"\n  [warmup] FAILED: {e}"); warmup_ok = False; break
        print("done")

        run_results = []
        print(f"  [bench]  {config.benchmark_runs} runs...", end=" ", flush=True)
        for i in range(config.benchmark_runs):
            try:
                r = pipeline.run_once(); r.run_id = i
                run_results.append(r)
                print(f"{i+1}", end=" ", flush=True)
            except Exception as e:
                print(f"\n  [bench] run {i} FAILED: {e}")
                run_results.append(StageTimings(
                    pipeline_name=pipeline.name, run_id=i, error=str(e)))
        print()
        all_results.extend(run_results)
        pipeline.teardown(); print("  [teardown] OK")
    return all_results

def save_csv(results, path):
    fields = ["pipeline_name","run_id","decode_s","preprocess_s","inference_s",
              "postprocess_s","total_s","frames_processed","fps","error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in results:
            row = asdict(r); row["fps"] = r.fps
            w.writerow({k: row.get(k,"") for k in fields})
    print(f"\n  Results saved → {path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video",      required=True)
    p.add_argument("--model",      required=True)
    p.add_argument("--batch",      type=int, default=8)
    p.add_argument("--warmup",     type=int, default=3)
    p.add_argument("--runs",       type=int, default=10)
    p.add_argument("--input-size", type=int, nargs=2, default=[640,640],
                   metavar=("W","H"))
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--no-fp16",    action="store_true")
    p.add_argument("--pipelines",  nargs="+",
                   choices=list(PIPELINE_REGISTRY.keys()),
                   default=list(PIPELINE_REGISTRY.keys()))
    p.add_argument("--out-csv",
                   default=str(RESULTS_DIR/"benchmark_results.csv"))
    args = p.parse_args()

    config = BenchmarkConfig(
        video_path=args.video, model_path=args.model,
        batch_size=args.batch, warmup_runs=args.warmup,
        benchmark_runs=args.runs, input_size=tuple(args.input_size),
        device=args.device, fp16=not args.no_fp16,
    )
    print(f"\nVideo Inference Pipeline Benchmark")
    for k,v in [("Video",config.video_path),("Model",config.model_path),
                ("Batch",config.batch_size),("Input",config.input_size),
                ("Device",config.device),("FP16",config.fp16),
                ("Pipelines",args.pipelines)]:
        print(f"  {k:<10}: {v}")

    results = run_benchmark(config, args.pipelines)
    save_csv(results, Path(args.out_csv))

    from collections import defaultdict
    print(f"\n{'─'*60}")
    print(f"  {'Pipeline':<30} {'Avg FPS':>9} {'Avg Total (s)':>14}")
    print(f"{'─'*60}")
    groups = defaultdict(list)
    for r in results:
        if not r.error: groups[r.pipeline_name].append(r)
    for pname, runs in groups.items():
        print(f"  {pname:<30} {sum(r.fps for r in runs)/len(runs):>9.1f} "
              f"{sum(r.total_s for r in runs)/len(runs):>14.3f}")
    print(f"{'─'*60}")

    from charts import generate_all_charts
    generate_all_charts(Path(args.out_csv), CHARTS_DIR)

if __name__ == "__main__":
    main()