"""
Chart generation for benchmark results.
Produces blog-ready PNG charts from the CSV output.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────

COLORS = {
    "OpenCV CPU":               "#8B8B8B",
    "FFmpeg + DALI":            "#4A90D9",
    "TorchCodec GPU":           "#7B68EE",
    "PyNvVideoCodec 2.1":       "#27AE60",
    "PyNvVideoCodec + CV-CUDA": "#E67E22",
}

STAGE_COLORS = {
    "decode":     "#5B9BD5",
    "preprocess": "#ED7D31",
    "inference":  "#A9D18E",
    "postprocess":"#9E6FCE",
}

STAGE_LABELS = {
    "decode":     "Decode",
    "preprocess": "Preprocess",
    "inference":  "Inference",
    "postprocess":"Postprocess",
}

def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.linestyle":   "--",
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.titleweight": "bold",
        "axes.labelsize":   11,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  10,
        "figure.dpi":       150,
    })


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            for k in ("decode_s","preprocess_s","inference_s","postprocess_s",
                      "total_s","fps","frames_processed","run_id"):
                try:
                    row[k] = float(row[k])
                except (ValueError, KeyError):
                    row[k] = 0.0
            rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> dict[str, dict]:
    """Return {pipeline_name: {metric: mean, metric_std: std, ...}}"""
    groups = defaultdict(list)
    for r in rows:
        groups[r["pipeline_name"]].append(r)

    result = {}
    for name, runs in groups.items():
        def mean(k): return np.mean([r[k] for r in runs])
        def std(k):  return np.std([r[k] for r in runs])
        result[name] = {
            "fps":            mean("fps"),
            "fps_std":        std("fps"),
            "total_s":        mean("total_s"),
            "total_std":      std("total_s"),
            "decode_s":       mean("decode_s"),
            "preprocess_s":   mean("preprocess_s"),
            "inference_s":    mean("inference_s"),
            "postprocess_s":  mean("postprocess_s"),
            "n_runs":         len(runs),
        }
    return result


# ── Chart 1 — FPS comparison (bar chart) ─────────────────────────────────────

def chart_fps_comparison(agg: dict, out: Path):
    _style()
    pipelines = list(agg.keys())
    fps_vals  = [agg[p]["fps"]     for p in pipelines]
    fps_errs  = [agg[p]["fps_std"] for p in pipelines]
    colors    = [COLORS.get(p, "#AAAAAA") for p in pipelines]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        range(len(pipelines)), fps_vals,
        yerr=fps_errs, capsize=5,
        color=colors, edgecolor="white", linewidth=0.8,
        error_kw={"ecolor": "#555555", "elinewidth": 1.2},
    )

    # value labels on top of bars
    for bar, val, err in zip(bars, fps_vals, fps_errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + err + max(fps_vals) * 0.01,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # speedup vs baseline (first pipeline)
    baseline = fps_vals[0] if fps_vals[0] > 0 else 1
    for i, (bar, val) in enumerate(zip(bars, fps_vals)):
        if i > 0 and baseline > 0:
            speedup = val / baseline
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(fps_vals) * 0.05,
                f"{speedup:.1f}×",
                ha="center", va="bottom",
                fontsize=9, color="white", fontweight="bold",
            )

    ax.set_xticks(range(len(pipelines)))
    ax.set_xticklabels(
        [p.replace(" + ", "\n+\n") for p in pipelines],
        fontsize=9,
    )
    ax.set_ylabel("Frames per second (FPS)")
    ax.set_title("End-to-end Throughput — Higher is Better")
    ax.set_ylim(0, max(fps_vals) * 1.18)

    fig.tight_layout()
    fig.savefig(out / "01_fps_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {out / '01_fps_comparison.png'}")


# ── Chart 2 — Per-stage stacked bar ──────────────────────────────────────────

def chart_stacked_stages(agg: dict, out: Path):
    _style()
    pipelines = list(agg.keys())
    stages    = ["decode_s", "preprocess_s", "inference_s", "postprocess_s"]
    stage_keys = ["decode", "preprocess", "inference", "postprocess"]

    x     = np.arange(len(pipelines))
    width = 0.55

    fig, ax = plt.subplots(figsize=(11, 6))
    bottoms = np.zeros(len(pipelines))

    for stage, key in zip(stages, stage_keys):
        vals = np.array([agg[p][stage] for p in pipelines])
        bars = ax.bar(
            x, vals, width,
            bottom=bottoms,
            label=STAGE_LABELS[key],
            color=STAGE_COLORS[key],
            edgecolor="white", linewidth=0.6,
        )
        # label segments > 5% of total
        for i, (bar, val) in enumerate(zip(bars, vals)):
            total = agg[pipelines[i]]["total_s"]
            if total > 0 and val / total > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottoms[i] + val / 2,
                    f"{val*1000:.0f}ms",
                    ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold",
                )
        bottoms += vals

    # total latency label above each bar
    for i, p in enumerate(pipelines):
        total = agg[p]["total_s"]
        ax.text(
            x[i], bottoms[i] + max(bottoms) * 0.01,
            f"{total*1000:.0f}ms",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [p.replace(" + ", "\n+\n") for p in pipelines], fontsize=9
    )
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Per-Stage Latency Breakdown — Lower is Better")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out / "02_stage_breakdown.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {out / '02_stage_breakdown.png'}")


# ── Chart 3 — Speedup waterfall vs OpenCV baseline ───────────────────────────

def chart_speedup_waterfall(agg: dict, out: Path):
    _style()
    pipelines = list(agg.keys())
    if not pipelines:
        return

    baseline_total = agg[pipelines[0]]["total_s"]
    if baseline_total == 0:
        return

    speedups = [baseline_total / agg[p]["total_s"] for p in pipelines]
    colors   = [COLORS.get(p, "#AAAAAA") for p in pipelines]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        range(len(pipelines)), speedups,
        color=colors, edgecolor="white", linewidth=0.8, height=0.55,
    )

    for bar, val in zip(bars, speedups):
        ax.text(
            val + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}×",
            va="center", ha="left", fontsize=10, fontweight="bold",
        )

    ax.axvline(1.0, color="#AAAAAA", linestyle="--", linewidth=1, label="Baseline (1×)")
    ax.set_yticks(range(len(pipelines)))
    ax.set_yticklabels(pipelines, fontsize=10)
    ax.set_xlabel("Speedup vs OpenCV CPU baseline")
    ax.set_title("End-to-end Speedup — Higher is Better")
    ax.set_xlim(0, max(speedups) * 1.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / "03_speedup_waterfall.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {out / '03_speedup_waterfall.png'}")


# ── Chart 4 — Batch sweep: CV-CUDA vs PyNvVideoCodec ─────────────────────────

def chart_batch_sweep(batch_csvs: dict, out: Path):
    """
    batch_csvs: {batch_size: csv_path} e.g. {16: Path(...), 32: Path(...)}
    Plots FPS for PyNvVideoCodec 2.1 vs PyNvVideoCodec + CV-CUDA across batch sizes.
    """
    _style()

    batch_sizes  = sorted(batch_csvs.keys())
    pynv_fps     = []
    cvcuda_fps   = []
    cvcuda_failed = []

    for b in batch_sizes:
        rows = load_csv(batch_csvs[b])
        agg  = aggregate(rows)
        pynv_fps.append(agg.get("PyNvVideoCodec 2.1", {}).get("fps", 0))
        cv = agg.get("PyNvVideoCodec + CV-CUDA", {}).get("fps", 0)
        if cv == 0:
            cvcuda_fps.append(None)
            cvcuda_failed.append(b)
        else:
            cvcuda_fps.append(cv)

    x     = np.arange(len(batch_sizes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width/2, pynv_fps, width,
                   label="PyNvVideoCodec 2.1",
                   color="#27AE60", edgecolor="white", linewidth=0.8)
    cv_plot = [v if v is not None else 0 for v in cvcuda_fps]
    bars2 = ax.bar(x + width/2, cv_plot, width,
                   label="PyNvVC + CV-CUDA",
                   color="#E67E22", edgecolor="white", linewidth=0.8)

    # value labels
    for bar, val in zip(bars1, pynv_fps):
        ax.text(bar.get_x() + bar.get_width()/2, val + 5,
                f"{val:.0f}", ha="center", fontsize=9, fontweight="bold", color="#27AE60")
    for bar, val, b in zip(bars2, cvcuda_fps, batch_sizes):
        if val is None:
            ax.text(bar.get_x() + bar.get_width()/2, 15,
                    "N/A", ha="center", fontsize=8, color="#999", style="italic")
        else:
            ax.text(bar.get_x() + bar.get_width()/2, val + 5,
                    f"{val:.0f}", ha="center", fontsize=9, fontweight="bold", color="#E67E22")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Batch {b}" for b in batch_sizes])
    ax.set_ylabel("Frames per second (FPS)")
    ax.set_title("CV-CUDA vs PyNvVideoCodec — FPS by Batch Size")
    ax.legend()
    ax.set_ylim(0, max(pynv_fps + [v for v in cvcuda_fps if v]) * 1.2)

    if cvcuda_failed:
        ax.annotate(
            f"CV-CUDA fails at batch {cvcuda_failed[0]}\n(tensor size limit)",
            xy=(batch_sizes.index(cvcuda_failed[0]) + width/2, 20),
            xytext=(batch_sizes.index(cvcuda_failed[0]) - 0.6, max(pynv_fps) * 0.6),
            fontsize=8, color="#999",
            arrowprops=dict(arrowstyle="->", color="#ccc"),
        )

    fig.tight_layout()
    fig.savefig(out / "04_batch_sweep.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {out / '04_batch_sweep.png'}")


# ── Chart 5 — Preprocess time comparison ─────────────────────────────────────

def chart_preprocess_comparison(agg: dict, out: Path):
    """Highlights the 23x preprocess reduction — the core insight of the benchmark."""
    _style()
    pipelines    = list(agg.keys())
    preprocess_ms = [agg[p]["preprocess_s"] * 1000 for p in pipelines]
    colors        = [COLORS.get(p, "#AAAAAA") for p in pipelines]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(pipelines)), preprocess_ms,
                  color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, preprocess_ms):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + max(preprocess_ms) * 0.01,
                f"{val:.1f}ms",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # speedup annotation vs baseline
    baseline = preprocess_ms[0] if preprocess_ms[0] > 0 else 1
    for i, (bar, val) in enumerate(zip(bars, preprocess_ms)):
        if i > 0 and val > 0:
            speedup = baseline / val
            ax.text(bar.get_x() + bar.get_width()/2,
                    max(preprocess_ms) * 0.06,
                    f"{speedup:.0f}×\nfaster",
                    ha="center", va="bottom",
                    fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(range(len(pipelines)))
    ax.set_xticklabels([p.replace(" + ", "\n+\n") for p in pipelines], fontsize=9)
    ax.set_ylabel("Preprocess time (ms, total across all batches)")
    ax.set_title("Preprocess Stage — CPU vs GPU  |  Lower is Better")
    ax.set_ylim(0, max(preprocess_ms) * 1.2)

    fig.tight_layout()
    fig.savefig(out / "05_preprocess_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {out / '05_preprocess_comparison.png'}")


# ── Entry point ───────────────────────────────────────────────────────────────

def generate_all_charts(csv_path: Path, out_dir: Path, batch_csvs: dict = None):
    """
    csv_path   : main benchmark CSV (all pipelines, single batch size)
    out_dir    : output directory for chart PNGs
    batch_csvs : optional dict {batch_size: Path} for the batch sweep chart
                 e.g. {16: Path('results/full_sweep_16.csv'), 32: Path(...), ...}
    """
    print(f"\n{'─'*60}")
    print(f"  Generating charts from {csv_path}")
    print(f"{'─'*60}")

    rows = load_csv(csv_path)
    if not rows:
        print("  No valid rows in CSV — skipping charts.")
        return

    agg = aggregate(rows)

    chart_fps_comparison(agg, out_dir)
    chart_stacked_stages(agg, out_dir)
    chart_speedup_waterfall(agg, out_dir)

    if batch_csvs:
        chart_batch_sweep(batch_csvs, out_dir)
    else:
        print("  Skipping batch sweep chart (no batch_csvs provided)")

    chart_preprocess_comparison(agg, out_dir)

    print(f"\n  All charts saved to {out_dir}")


if __name__ == "__main__":
    import sys
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/benchmark_results.csv")
    out_dir  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("charts")

    # Auto-detect batch sweep CSVs if they exist alongside the main CSV
    results_dir = csv_path.parent
    batch_csvs = {}
    for b in [16, 32, 64, 128]:
        p = results_dir / f"full_sweep_{b}.csv"
        if p.exists():
            batch_csvs[b] = p
    if not batch_csvs:
        # also try the old naming pattern
        for b in [16, 32, 64, 128]:
            p = results_dir / f"batch_sweep_{b}.csv"
            if p.exists():
                batch_csvs[b] = p

    generate_all_charts(csv_path, out_dir, batch_csvs or None)