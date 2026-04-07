# Video Inference Pipeline Benchmark

End-to-end benchmark comparing 5 video inference pipeline configurations
across decode → preprocess → inference → postprocess.

## Pipelines Compared

| # | Pipeline | Decoder | Preprocessor | Notes |
|---|----------|---------|--------------|-------|
| 1 | **OpenCV CPU** | OpenCV CPU | OpenCV CPU | Baseline |
| 2 | **FFmpeg + DALI** | NVDEC via DALI | DALI GPU kernels | Article reference |
| 3 | **TorchCodec GPU** | NVDEC via TorchCodec | torch.nn.functional | Meta/PyTorch native |
| 4 | **PyNvVideoCodec 2.1** | NVDEC ThreadedDecoder | torch.nn.functional | NVIDIA official, background prefetch |
| 5 | **PyNvVideoCodec + CV-CUDA** | NVDEC ThreadedDecoder | CV-CUDA GPU ops | Zero-copy, max throughput |

All pipelines use the same YOLO model for inference and PyTorch for postprocessing.

---

## Requirements

- NVIDIA GPU with NVDEC (A100/H100/V100 recommended)
- CUDA 12.x
- Python 3.10+

## Installation

```bash
# Core
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics matplotlib numpy

# DALI (Pipeline 2)
pip install nvidia-dali-cuda120

# TorchCodec (Pipeline 3)
pip install torchcodec --index-url https://download.pytorch.org/whl/cu124

# PyNvVideoCodec 2.1 (Pipelines 4 & 5)
pip install pynvvideocodec

# CV-CUDA (Pipeline 5)
pip install cvcuda-cu12
```

## Quick Start

### Dry run (no GPU required — validates harness with synthetic data)
```bash
python dry_run.py
```

### Real benchmark
```bash
python benchmark.py \
  --video  /path/to/video.mp4 \
  --model  yolov8n.pt \
  --batch  8 \
  --runs   10 \
  --warmup 3
```

### Run only specific pipelines
```bash
python benchmark.py \
  --video  /path/to/video.mp4 \
  --model  yolov8n.pt \
  --pipelines opencv_cpu ffmpeg_dali torchcodec pynvvideocodec pynv_cvcuda
```

### Generate charts from existing CSV
```bash
python charts.py results/benchmark_results.csv charts/
```

---

## Output

```
video_benchmark/
├── results/
│   └── benchmark_results.csv       # Per-run timings for all pipelines
└── charts/
    ├── 01_fps_comparison.png        # FPS bar chart with speedup labels
    ├── 02_stage_breakdown.png       # Stacked bar: decode/preprocess/infer/postprocess
    ├── 03_speedup_waterfall.png     # Horizontal speedup vs baseline
    ├── 04_latency_distribution.png  # Box plot across runs
    └── 05_decode_analysis.png       # Decode time absolute + % of total
```

## CSV Schema

| Column | Description |
|--------|-------------|
| `pipeline_name` | Human-readable pipeline name |
| `run_id` | Run index (0-based) |
| `decode_s` | Decode stage time (seconds) |
| `preprocess_s` | Preprocess stage time (seconds) |
| `inference_s` | Inference stage time (seconds) |
| `postprocess_s` | Postprocess stage time (seconds) |
| `total_s` | End-to-end wall time (seconds) |
| `frames_processed` | Number of frames in this run |
| `fps` | `frames_processed / total_s` |
| `error` | Error message if run failed, else empty |

---

## Tips for A100/H100

- A100 has **5 NVDEC engines** — use `--batch 16` or higher to saturate them
- For H100, try `--batch 32` and `num_streams=8` in config
- TensorRT FP16 engine for YOLO gives another ~2× on inference time
  - Export: `yolo export model=yolov8n.pt format=engine half=True`
  - Then pass `--model yolov8n.engine` to the benchmark

## Adding a New Pipeline

1. Subclass `BasePipeline` in `benchmark.py`
2. Implement `setup()`, `run_once()`, `teardown()`
3. Add an entry to `PIPELINE_REGISTRY`
4. Add a latency profile entry to `dry_run.py` for dry-run testing
