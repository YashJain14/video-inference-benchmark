"""
dry_run.py — Validates the full benchmark harness with synthetic data.
No real GPU libraries required. Useful for CI or pre-flight testing.

Usage:
    python dry_run.py
"""

import csv
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
CHARTS_DIR  = Path(__file__).parent / "charts"
RESULTS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

# Simulated latency profiles (seconds) based on expected real-world ranges
# on an A100 for a 1-min 1080p video, batch=8, YOLO inference
LATENCY_PROFILES = {
    "OpenCV CPU": {
        "decode_s":      (2.8,  3.2),
        "preprocess_s":  (1.8,  2.2),
        "inference_s":   (8.5,  9.5),
        "postprocess_s": (0.8,  1.0),
    },
    "FFmpeg + DALI": {
        "decode_s":      (0.8,  1.0),    # NVDEC
        "preprocess_s":  (0.3,  0.5),    # DALI GPU kernels
        "inference_s":   (1.1,  1.5),    # PyTorch GPU
        "postprocess_s": (0.1,  0.2),
    },
    "TorchCodec GPU": {
        "decode_s":      (0.6,  0.9),    # NVDEC via TorchCodec
        "preprocess_s":  (0.25, 0.40),   # torch.nn.functional
        "inference_s":   (1.0,  1.4),
        "postprocess_s": (0.09, 0.15),
    },
    "PyNvVideoCodec 2.1": {
        "decode_s":      (0.45, 0.65),   # ThreadedDecoder hides latency
        "preprocess_s":  (0.22, 0.38),
        "inference_s":   (0.90, 1.20),
        "postprocess_s": (0.08, 0.13),
    },
    "PyNvVideoCodec + CV-CUDA": {
        "decode_s":      (0.45, 0.65),
        "preprocess_s":  (0.10, 0.18),   # CV-CUDA zero-copy GPU ops
        "inference_s":   (0.85, 1.10),
        "postprocess_s": (0.07, 0.11),
    },
}

FRAMES_PROCESSED = 8  # matches default batch_size


def simulate_run(pipeline_name: str, run_id: int) -> dict:
    profile = LATENCY_PROFILES[pipeline_name]
    row = {
        "pipeline_name":   pipeline_name,
        "run_id":          run_id,
        "frames_processed": FRAMES_PROCESSED,
        "error":           "",
    }
    total = 0.0
    for stage in ("decode_s", "preprocess_s", "inference_s", "postprocess_s"):
        lo, hi = profile[stage]
        val = random.uniform(lo, hi)
        row[stage] = round(val, 6)
        total += val
    row["total_s"] = round(total, 6)
    row["fps"]     = round(FRAMES_PROCESSED / total, 2)
    return row


def main():
    WARMUP_RUNS    = 3
    BENCHMARK_RUNS = 10
    all_rows       = []

    print("\nDry run — simulating benchmark with synthetic latency data")
    print("(No GPU libraries required)\n")

    for pipeline_name in LATENCY_PROFILES:
        print(f"  Pipeline: {pipeline_name}")
        print(f"    warmup {WARMUP_RUNS} runs... ", end="", flush=True)
        for _ in range(WARMUP_RUNS):
            simulate_run(pipeline_name, -1)
            time.sleep(0.01)
        print("done")

        print(f"    bench  {BENCHMARK_RUNS} runs... ", end="", flush=True)
        for i in range(BENCHMARK_RUNS):
            row = simulate_run(pipeline_name, i)
            all_rows.append(row)
            print(f"{i+1}", end=" ", flush=True)
        print()

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    fieldnames = [
        "pipeline_name", "run_id",
        "decode_s", "preprocess_s", "inference_s", "postprocess_s",
        "total_s", "frames_processed", "fps", "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Results saved → {csv_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    from collections import defaultdict
    import numpy as np

    groups = defaultdict(list)
    for r in all_rows:
        groups[r["pipeline_name"]].append(r)

    print(f"\n{'─'*70}")
    print(f"  {'Pipeline':<32} {'Avg FPS':>8} {'Avg Total (ms)':>15} {'Speedup':>8}")
    print(f"{'─'*70}")
    baseline_ms = None
    for pname, runs in groups.items():
        avg_fps   = np.mean([r["fps"]     for r in runs])
        avg_total = np.mean([r["total_s"] for r in runs]) * 1000
        if baseline_ms is None:
            baseline_ms = avg_total
        speedup = baseline_ms / avg_total
        print(f"  {pname:<32} {avg_fps:>8.1f} {avg_total:>15.1f} {speedup:>7.2f}×")
    print(f"{'─'*70}")

    # ── Generate charts ───────────────────────────────────────────────────────
    from charts import generate_all_charts
    generate_all_charts(csv_path, CHARTS_DIR)
    print(f"\n  Charts saved → {CHARTS_DIR}")


if __name__ == "__main__":
    main()
