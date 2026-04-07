# Video Inference Pipeline Benchmark

A benchmarking harness for GPU-accelerated video inference pipelines, comparing four
configurations end-to-end across decode → preprocess → inference → postprocess on
NVIDIA data centre GPUs.

**Hardware:** NVIDIA A100-SXM4-40GB · CUDA 12.4 · PyTorch 2.6  
**Results:** [Blog post](https://yashjain14.github.io/blogs/pynvvideocodec-vs-dali-vs-opencv-benchmark.html)

---

## Results (A100-SXM4-40GB, YOLOv8n, 1024 frames, batch 16)

| Pipeline | Avg FPS | Total (s) | Decode (s) | Preprocess (s) | Inference (s) | Speedup |
|---|---|---|---|---|---|---|
| OpenCV CPU | 140 | 7.303 | 1.752 | 2.471 | 3.065 | 1× |
| FFmpeg + DALI | 383 | 2.675 | 1.644 | 0.107 | 0.910 | **2.7×** |
| PyNvVideoCodec 2.1 | 405 | 2.528 | 1.497 | 0.107 | 0.910 | **2.9×** |
| PyNvVideoCodec + CV-CUDA | 451 | 2.270 | 1.444 | 0.086 | 0.728 | **3.2×** |

### Batch sweep

| Pipeline | B16 FPS | B32 FPS | B64 FPS | B128 FPS |
|---|---|---|---|---|
| OpenCV CPU | 140.2 | 141.9 | 144.4 | 148.8 |
| FFmpeg + DALI | 382.8 | 387.7 | 391.3 | 396.7 |
| PyNvVideoCodec 2.1 | 405.0 | 414.0 | 427.3 | 420.4 |
| PyNvVideoCodec + CV-CUDA | 451.1 | 455.0 | 459.3 | ❌ crash |

> CV-CUDA fails at batch 128 with `NVCV_ERROR_INVALID_ARGUMENT: Input or output tensors are too large`.
> PyNvVideoCodec peaks at batch 64 (427 FPS) and regresses slightly to 420 FPS at batch 128.

---

## Pipelines

| Key | Pipeline | Decoder | Preprocessor |
|---|---|---|---|
| `opencv_cpu` | OpenCV CPU | `cv2.VideoCapture` | `cv2.resize` on CPU |
| `ffmpeg_dali` | FFmpeg + DALI | NVDEC via `fn.experimental.readers.video` | `torch.nn.functional` |
| `pynvvideocodec` | PyNvVideoCodec 2.1 | NVDEC via `ThreadedDecoder` | `torch.nn.functional` |
| `pynv_cvcuda` | PyNvVideoCodec + CV-CUDA | NVDEC via `ThreadedDecoder` | `cvcuda.resize` + `cvcuda.convertto` |

All pipelines use the same YOLOv8n model (FP16) and PyTorch for postprocessing.

---

## Benchmarking Methodology

Each pipeline **pre-decodes `total_frames` once at `setup()`** and caches raw frames
in GPU memory. The timed benchmark loop measures only preprocess → inference →
postprocess. Decode time is captured separately as a single timed pass and added back
into `total_s`. This ensures:

- Every pipeline processes the **same frames** from the same video positions
- File I/O overhead does not contaminate benchmark runs
- Decode time is still reported honestly — once per pipeline
- FPS numbers reflect steady-state throughput on identical workloads

---

## Requirements

- NVIDIA GPU with NVDEC hardware (A100 / H100 / V100 recommended)
- CUDA 12.x
- Python 3.10+
- Linux (CV-CUDA does not support Windows natively)

---

## Installation

```bash
# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics matplotlib numpy

# FFmpeg + DALI (pipeline: ffmpeg_dali)
pip install nvidia-dali-cuda120
conda install -c conda-forge ffmpeg -y   # needed for video conversion

# PyNvVideoCodec 2.1 (pipelines: pynvvideocodec, pynv_cvcuda)
pip install pynvvideocodec

# CV-CUDA (pipeline: pynv_cvcuda only)
pip install cvcuda-cu12
```

> **Note on GPU libraries (HPC clusters):** If `PyNvVideoCodec` fails to import with
> `libnvidia-encode.so.1: cannot open shared object file`, add the driver lib path:
> ```bash
> export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
> ```

---

## Video Preparation

DALI's video reader requires an MP4 file with the index at the front (`faststart`).
If your source is a `.mov` file, convert first:

```bash
ffmpeg -i input.mov -c:v copy -c:a copy -movflags +faststart output.mp4
```

---

## Usage

### Run the full benchmark

```bash
python benchmark.py \
  --video  ToS.mp4 \
  --model  yolov8n.pt \
  --total-frames 1024 \
  --batch  16 \
  --runs   10 \
  --warmup 3 \
  --pipelines opencv_cpu ffmpeg_dali pynvvideocodec pynv_cvcuda
```

### Run a single pipeline

```bash
python benchmark.py \
  --video ToS.mp4 \
  --model yolov8n.pt \
  --pipelines ffmpeg_dali \
  --runs 3 --warmup 1
```

### Batch sweep (reproduces blog results)

```bash
for BATCH in 16 32 64 128; do
  python benchmark.py \
    --video ToS.mp4 \
    --model yolov8n.pt \
    --total-frames 1024 \
    --batch $BATCH \
    --runs 10 \
    --warmup 3 \
    --pipelines opencv_cpu ffmpeg_dali pynvvideocodec pynv_cvcuda \
    --out-csv results/full_sweep_$BATCH.csv
done
```

### Regenerate charts from an existing CSV

```bash
python charts.py results/benchmark_results.csv charts/
```

---

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--video` | required | Path to input video file (MP4 recommended) |
| `--model` | required | Path to YOLO model (`.pt` or TensorRT `.engine`) |
| `--total-frames` | `512` | Fixed frame count every pipeline processes per run |
| `--batch` | `16` | Frames per inference batch |
| `--warmup` | `3` | Warmup runs before timing |
| `--runs` | `10` | Timed benchmark runs |
| `--input-size` | `640 640` | Model input resolution (W H) |
| `--device` | `cuda:0` | PyTorch device |
| `--no-fp16` | off | Disable FP16 inference |
| `--pipelines` | all | Space-separated list of pipeline keys to run |
| `--out-csv` | `results/benchmark_results.csv` | Output CSV path |

---

## Output

```
video-inference-benchmark/
├── benchmark.py              # Main harness — all 4 pipeline classes
├── charts.py                 # Chart generation from CSV
├── run_benchmark.pbs         # PBS job script for HPC clusters
├── results/
│   └── benchmark_results.csv # Per-run timings for all pipelines
└── charts/
    ├── 01_fps_comparison.png       # FPS bar chart with speedup labels
    ├── 02_stage_breakdown.png      # Stacked bar: decode/preprocess/infer/post
    ├── 03_speedup_waterfall.png    # Horizontal speedup vs CPU baseline
    ├── 04_latency_distribution.png # Box plot across 10 runs
    └── 05_decode_analysis.png      # Decode time absolute + % of total
```

### CSV schema

| Column | Description |
|---|---|
| `pipeline_name` | Pipeline identifier |
| `run_id` | Run index (0-based) |
| `decode_s` | Decode time in seconds (single timed pass at setup) |
| `preprocess_s` | Cumulative preprocess time across all batches |
| `inference_s` | Cumulative inference time across all batches |
| `postprocess_s` | Cumulative postprocess time across all batches |
| `total_s` | End-to-end wall time including decode |
| `frames_processed` | Total frames processed this run |
| `fps` | `frames_processed / total_s` |
| `error` | Error message if run failed, else empty |

---

## Extending the Benchmark

To add a new pipeline:

1. Subclass `BasePipeline` in `benchmark.py`
2. Implement `setup()` — decode frames once, store in `self._raw_frames`
3. Implement `run_once()` — loop over cached frames, accumulate stage timings
4. Register in `PIPELINE_REGISTRY` with a short key

```python
class MyPipeline(BasePipeline):
    name = "My Pipeline"

    def setup(self):
        self._model = self._load_yolo_model()
        # decode once → self._raw_frames, self._decode_time

    def run_once(self) -> StageTimings:
        t = self._make_timings(0)
        with Timer() as total:
            for batch in self._raw_frames:
                with Timer() as tp: ...   # preprocess
                t.preprocess_s += tp.elapsed
                with Timer() as ti: ...   # inference
                t.inference_s += ti.elapsed
            t.frames_processed = ...
        t.total_s = total.elapsed + t.decode_s
        return t
```

---

## Known Issues

- **CV-CUDA double normalization:** The `pynv_cvcuda` pipeline currently divides by
  255 twice — once with `.float().div(255.0)` before `cvcuda.resize`, and again with
  `cvcuda.convertto(..., scale=1.0/255.0)`. This means the model receives pixel values
  in the range [0, 0.0039] instead of [0, 1]. The throughput numbers are still valid
  (the GPU does the same amount of work regardless of pixel values), but detection
  accuracy would be affected in production. Fix: remove the `.float().div(255.0)` and
  pass uint8 data directly to CV-CUDA, or set `scale=1.0` in `convertto`.

- **TorchCodec** requires `libnppicc.so.12` which is not available on some HPC clusters.
  Install with `conda install -c nvidia cuda-nppc` if needed.

- **CV-CUDA** crashes at batch 128 with `NVCV_ERROR_INVALID_ARGUMENT: Input or output
  tensors are too large`. Use PyNvVideoCodec with `torch.nn.functional.interpolate` at
  batch sizes ≥ 128.

- **DALI** requires MP4 with `faststart` flag. MOV containers cause slow seeking
  and dramatically inflated decode times.

- All pipelines use sequential inference. Adding multi-stream CUDA parallelism
  and TensorRT FP16 are the next logical optimizations.