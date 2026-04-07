"""
Video Inference Pipeline Benchmark
====================================
Fair, fast benchmarking strategy:
  - Each pipeline pre-decodes total_frames ONCE at setup() into GPU memory
  - The benchmark loop only times: preprocess → inference → postprocess
  - Decode is measured separately as a single timed pass
  - All pipelines process identical frames from the same video positions

Pipelines:
  1. OpenCV CPU          (baseline)
  2. FFmpeg + DALI       (article baseline)
  3. TorchCodec GPU      (skipped if libnppicc missing)
  4. PyNvVideoCodec 2.1  (ThreadedDecoder)
  5. PyNvVideoCodec 2.1  + CV-CUDA preprocessing
"""

import argparse
import csv
import gc
import math
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

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
    total_frames: int   = 512   # identical for all pipelines
    batch_size: int     = 16
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

    def __init__(self, config):
        self.config = config
        self._model = None
        # Cached raw frames: list of [H, W, C] uint8 GPU tensors
        # Populated once in setup(), reused every run_once()
        self._raw_frames: list = []
        self._decode_time: float = 0.0  # single decode pass time

    def setup(self): raise NotImplementedError
    def run_once(self) -> StageTimings: raise NotImplementedError

    def teardown(self):
        self._model = None
        self._raw_frames = []
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _load_yolo_model(self):
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

    def _num_batches(self):
        return math.ceil(self.config.total_frames / self.config.batch_size)

    def _make_timings(self, run_id: int) -> StageTimings:
        """Start a new StageTimings with decode_s pre-filled from setup."""
        t = StageTimings(pipeline_name=self.name, run_id=run_id)
        t.decode_s = self._decode_time   # decode measured once at setup
        return t

# ─────────────────────────────────────────────
# Pipeline 1 — OpenCV CPU
# ─────────────────────────────────────────────

class OpenCVCPUPipeline(BasePipeline):
    name = "OpenCV CPU"

    def setup(self):
        import cv2; self._cv2 = cv2
        self._model = self._load_yolo_model()
        cfg = self.config

        # Decode total_frames once, store as CPU numpy (OpenCV native)
        with Timer() as td:
            cap = cv2.VideoCapture(cfg.video_path)
            frames = []
            while len(frames) < cfg.total_frames:
                ret, f = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, f = cap.read()
                    if not ret: break
                frames.append(f)
            cap.release()
        self._decode_time = td.elapsed
        self._raw_frames  = frames
        print(f"    decoded {len(frames)} frames in {td.elapsed:.3f}s")

    def run_once(self):
        cv2, cfg = self._cv2, self.config
        B = cfg.batch_size
        t = self._make_timings(0)

        with Timer() as total:
            processed = 0
            for b in range(self._num_batches()):
                chunk = self._raw_frames[b*B : b*B + B]
                if not chunk: break

                with Timer() as tp:
                    batch = []
                    for f in chunk:
                        r = cv2.resize(f, cfg.input_size)
                        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
                        batch.append(
                            torch.from_numpy(r).float().div(255.0).permute(2,0,1)
                        )
                    batch_tensor = torch.stack(batch)
                t.preprocess_s += tp.elapsed

                with Timer() as ti: results = self._infer(batch_tensor)
                t.inference_s += ti.elapsed

                with Timer() as tp2: [self._postprocess(r) for r in results]
                t.postprocess_s += tp2.elapsed

                processed += len(chunk)

            t.frames_processed = processed

        t.total_s = total.elapsed + t.decode_s
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
        self._build_decode_pipe()
        self._build_preprocess_pipe()
        self._decode_frames()

    def _build_decode_pipe(self):
        """
        Raw decode pipeline using fn.experimental.readers.video.

        Key fixes vs legacy fn.readers.video:
          - experimental reader has better seeking on MP4 (no full-file seeks)
          - batch_size=1, sequence_length=B: DALI's natural usage for single-video
            inference — reads B consecutive frames per iteration, one seek per batch.
            This avoids the B-seeks-per-batch problem of sequence_length=1.
          - Requires MP4 with faststart (moov atom at front) for fast seeks.
            Convert: ffmpeg -i input.mov -c copy -movflags +faststart output.mp4
        """
        fn, dali, cfg = self._fn, self._dali, self.config
        B = cfg.batch_size

        # Use MP4 path — swap .mov -> .mp4 if needed
        mp4_path = cfg.video_path.replace(".mov", ".mp4")

        @self._pd(batch_size=1, num_threads=4, device_id=0)
        def decode_pipe():
            # batch_size=1, sequence_length=B -> shape [1, B, H, W, C]
            frames = fn.experimental.readers.video(
                filenames=[mp4_path],
                sequence_length=B,
                normalized=False,
                random_shuffle=False,
                image_type=dali.types.RGB,
                name="Reader",
                device="gpu",
            )
            return frames

        self._decode_pipe = decode_pipe()
        self._decode_pipe.build()

    def _build_preprocess_pipe(self):
        """Not used — preprocess runs in run_once via torch ops for fair timing."""
        pass

    def _decode_frames(self):
        """
        Decode total_frames once via single-pass streaming.
        Each pipe.run() returns [1, B, H, W, C] — squeeze to [B, H, W, C].
        """
        cfg = self.config
        n_batches = self._num_batches()
        frames = []

        with Timer() as td:
            for _ in range(n_batches):
                out = self._decode_pipe.run()
                dt  = out[0].as_tensor()
                bt  = torch.empty(dt.shape(), dtype=torch.uint8, device=cfg.device)
                bt.copy_(torch.as_tensor(dt, device=cfg.device))
                # shape: [1, B, H, W, C] -> squeeze batch dim -> [B, H, W, C]
                if bt.ndim == 5:
                    bt = bt.squeeze(0)
                frames.append(bt)

        self._decode_time = td.elapsed
        self._raw_frames  = frames
        print(f"    decoded {n_batches} batches in {td.elapsed:.3f}s")

    def run_once(self):
        cfg = self.config
        t   = self._make_timings(0)

        with Timer() as total:
            for batch_gpu in self._raw_frames:
                with Timer() as tp:
                    # resize + normalize on GPU via torch (DALI preprocess
                    # requires external_source feed which adds overhead;
                    # use torch ops directly for fair per-batch timing)
                    f = batch_gpu.float().div(255.0).permute(0,3,1,2).contiguous()
                    f = F.interpolate(f, size=cfg.input_size,
                                      mode="bilinear", align_corners=False)
                t.preprocess_s += tp.elapsed

                with Timer() as ti: results = self._infer(f)
                t.inference_s += ti.elapsed

                with Timer() as tp2: [self._postprocess(r) for r in results]
                t.postprocess_s += tp2.elapsed

            t.frames_processed = len(self._raw_frames) * cfg.batch_size

        t.total_s = total.elapsed + t.decode_s
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
            raise RuntimeError(f"TorchCodec failed: {e}\n"
                               "Fix: conda install -c nvidia cuda-nppc")
        self._model = self._load_yolo_model()
        self._decode_frames()

    def _decode_frames(self):
        cfg = self.config
        B   = cfg.batch_size

        with Timer() as td:
            dec     = self._VideoDecoder(cfg.video_path, device=cfg.device)
            n_video = len(dec)
            frames  = []
            for b in range(self._num_batches()):
                start = (b * B) % n_video
                end   = min(start + B, n_video)
                fb    = dec.get_frames_at(indices=list(range(start, end)))
                frames.append(fb.data.clone())   # [B, C, H, W] uint8 on GPU

        self._decode_time = td.elapsed
        self._raw_frames  = frames
        print(f"    decoded {len(frames)} batches in {td.elapsed:.3f}s")

    def run_once(self):
        cfg = self.config
        t   = self._make_timings(0)

        with Timer() as total:
            for fgpu in self._raw_frames:
                with Timer() as tp:
                    f = fgpu.float().div(255.0)
                    f = F.interpolate(f, size=cfg.input_size,
                                      mode="bilinear", align_corners=False)
                t.preprocess_s += tp.elapsed

                with Timer() as ti: results = self._infer(f)
                t.inference_s += ti.elapsed

                with Timer() as tp2: [self._postprocess(r) for r in results]
                t.postprocess_s += tp2.elapsed

            t.frames_processed = sum(f.shape[0] for f in self._raw_frames)

        t.total_s = total.elapsed + t.decode_s
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
        self._decode_frames()

    def _decode_frames(self):
        """Decode total_frames ONCE via a single ThreadedDecoder instance."""
        cfg = self.config
        N   = cfg.total_frames
        B   = cfg.batch_size

        with Timer() as td:
            # Single decoder open — reads all frames in one pass
            dec   = self._TD(cfg.video_path, N * 2, gpu_id=0,
                             output_color_type=self._OCT.RGB)
            all_f = dec.get_batch_frames(N)
            dec.end()

            # Split into batches and store as GPU tensors
            frames = []
            for i in range(0, len(all_f), B):
                chunk = all_f[i : i + B]
                batch = torch.stack([
                    torch.as_tensor(f, device=cfg.device).clone() for f in chunk
                ])  # [B, H, W, C] uint8
                frames.append(batch)

        self._decode_time = td.elapsed
        self._raw_frames  = frames
        print(f"    decoded {N} frames in {td.elapsed:.3f}s")

    def run_once(self):
        cfg = self.config
        t   = self._make_timings(0)

        with Timer() as total:
            for batch_hwc in self._raw_frames:
                with Timer() as tp:
                    # [B,H,W,C] uint8 -> [B,C,H,W] float32 -> resize
                    f = batch_hwc.float().div(255.0).permute(0,3,1,2).contiguous()
                    f = F.interpolate(f, size=cfg.input_size,
                                      mode="bilinear", align_corners=False)
                t.preprocess_s += tp.elapsed

                with Timer() as ti: results = self._infer(f)
                t.inference_s += ti.elapsed

                with Timer() as tp2: [self._postprocess(r) for r in results]
                t.postprocess_s += tp2.elapsed

            t.frames_processed = sum(b.shape[0] for b in self._raw_frames)

        t.total_s = total.elapsed + t.decode_s
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
        self._decode_frames()

    def _decode_frames(self):
        """Same single-pass decode as Pipeline 4."""
        cfg = self.config
        N, B = cfg.total_frames, cfg.batch_size

        with Timer() as td:
            dec   = self._TD(cfg.video_path, N*2, gpu_id=0,
                             output_color_type=self._OCT.RGB)
            all_f = dec.get_batch_frames(N)
            dec.end()
            frames = []
            for i in range(0, len(all_f), B):
                chunk = all_f[i : i + B]
                batch = torch.stack([
                    torch.as_tensor(f, device=cfg.device).clone() for f in chunk
                ])
                frames.append(batch)

        self._decode_time = td.elapsed
        self._raw_frames  = frames
        print(f"    decoded {N} frames in {td.elapsed:.3f}s")

    def run_once(self):
        cfg, cvcuda = self.config, self._cvcuda
        t = self._make_timings(0)

        with Timer() as total:
            for batch_hwc in self._raw_frames:
                with Timer() as tp:
                    # [B,H,W,C] uint8 -> float32 NHWC for CV-CUDA
                    nhwc      = batch_hwc.float().div(255.0).contiguous()
                    cvcuda_in = cvcuda.as_tensor(nhwc, "NHWC")
                    rsz       = cvcuda.resize(
                        cvcuda_in,
                        (nhwc.shape[0], cfg.input_size[1], cfg.input_size[0], 3),
                        cvcuda.Interp.LINEAR,
                    )
                    nrm   = cvcuda.convertto(rsz, cvcuda.Type.F32,
                                             scale=1.0/255.0, offset=0.0)
                    final = torch.as_tensor(
                        nrm.cuda(), device=cfg.device
                    ).permute(0,3,1,2).contiguous()
                t.preprocess_s += tp.elapsed

                with Timer() as ti: results = self._infer(final)
                t.inference_s += ti.elapsed

                with Timer() as tp2: [self._postprocess(r) for r in results]
                t.postprocess_s += tp2.elapsed

            t.frames_processed = sum(b.shape[0] for b in self._raw_frames)

        t.total_s = total.elapsed + t.decode_s
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
        for _ in range(config.warmup_runs):
            try: pipeline.run_once()
            except Exception as e: print(f"\n  [warmup] FAILED: {e}"); break
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
    p.add_argument("--video",        required=True)
    p.add_argument("--model",        required=True)
    p.add_argument("--total-frames", type=int, default=512)
    p.add_argument("--batch",        type=int, default=16)
    p.add_argument("--warmup",       type=int, default=3)
    p.add_argument("--runs",         type=int, default=10)
    p.add_argument("--input-size",   type=int, nargs=2, default=[640,640],
                   metavar=("W","H"))
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--no-fp16",      action="store_true")
    p.add_argument("--pipelines",    nargs="+",
                   choices=list(PIPELINE_REGISTRY.keys()),
                   default=list(PIPELINE_REGISTRY.keys()))
    p.add_argument("--out-csv",
                   default=str(RESULTS_DIR/"benchmark_results.csv"))
    args = p.parse_args()

    config = BenchmarkConfig(
        video_path=args.video, model_path=args.model,
        total_frames=args.total_frames,
        batch_size=args.batch, warmup_runs=args.warmup,
        benchmark_runs=args.runs, input_size=tuple(args.input_size),
        device=args.device, fp16=not args.no_fp16,
    )
    print(f"\nVideo Inference Pipeline Benchmark")
    for k,v in [("Video",config.video_path),("Model",config.model_path),
                ("Total frames",config.total_frames),("Batch",config.batch_size),
                ("Input",config.input_size),("Device",config.device),
                ("FP16",config.fp16),("Pipelines",args.pipelines)]:
        print(f"  {k:<14}: {v}")

    results = run_benchmark(config, args.pipelines)
    save_csv(results, Path(args.out_csv))

    from collections import defaultdict
    print(f"\n{'─'*60}")
    print(f"  {'Pipeline':<30} {'Avg FPS':>9} {'Avg Total (s)':>14} {'Frames':>8}")
    print(f"{'─'*60}")
    groups = defaultdict(list)
    for r in results:
        if not r.error: groups[r.pipeline_name].append(r)
    for pname, runs in groups.items():
        avg_fps    = sum(r.fps for r in runs) / len(runs)
        avg_total  = sum(r.total_s for r in runs) / len(runs)
        avg_frames = sum(r.frames_processed for r in runs) / len(runs)
        print(f"  {pname:<30} {avg_fps:>9.1f} {avg_total:>14.3f} {avg_frames:>8.0f}")
    print(f"{'─'*60}")

    from charts import generate_all_charts
    generate_all_charts(Path(args.out_csv), CHARTS_DIR)

if __name__ == "__main__":
    main()