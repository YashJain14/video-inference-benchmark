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


# ── Chart 4 — Per-run latency distribution (box plot) ────────────────────────

def chart_latency_distribution(rows: list[dict], out: Path):
    _style()
    groups = defaultdict(list)
    for r in rows:
        groups[r["pipeline_name"]].append(r["total_s"] * 1000)  # ms

    pipelines = list(groups.keys())
    data      = [groups[p] for p in pipelines]
    colors    = [COLORS.get(p, "#AAAAAA") for p in pipelines]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_xticks(range(1, len(pipelines) + 1))
    ax.set_xticklabels(
        [p.replace(" + ", "\n+\n") for p in pipelines], fontsize=9
    )
    ax.set_ylabel("End-to-end latency (ms)")
    ax.set_title("Latency Distribution Across Runs")

    fig.tight_layout()
    fig.savefig(out / "04_latency_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {out / '04_latency_distribution.png'}")


# ── Chart 5 — Decode-only comparison (% of total) ────────────────────────────

def chart_decode_share(agg: dict, out: Path):
    _style()
    pipelines = list(agg.keys())
    shares = [
        100 * agg[p]["decode_s"] / agg[p]["total_s"]
        if agg[p]["total_s"] > 0 else 0
        for p in pipelines
    ]
    decode_ms = [agg[p]["decode_s"] * 1000 for p in pipelines]
    colors    = [COLORS.get(p, "#AAAAAA") for p in pipelines]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # left: decode time (ms)
    ax1.bar(range(len(pipelines)), decode_ms, color=colors, edgecolor="white")
    for i, val in enumerate(decode_ms):
        ax1.text(i, val + max(decode_ms) * 0.01, f"{val:.0f}ms",
                 ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(range(len(pipelines)))
    ax1.set_xticklabels(
        [p.replace(" + ", "\n+\n") for p in pipelines], fontsize=8
    )
    ax1.set_ylabel("Decode time (ms)")
    ax1.set_title("Decode Stage — Absolute Time")

    # right: decode % of total
    ax2.bar(range(len(pipelines)), shares, color=colors, edgecolor="white")
    for i, val in enumerate(shares):
        ax2.text(i, val + 1, f"{val:.1f}%",
                 ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(range(len(pipelines)))
    ax2.set_xticklabels(
        [p.replace(" + ", "\n+\n") for p in pipelines], fontsize=8
    )
    ax2.set_ylabel("Decode as % of total pipeline time")
    ax2.set_title("Decode Stage — Share of Total")
    ax2.set_ylim(0, 100)

    fig.suptitle("Decode Stage Analysis", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "05_decode_analysis.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {out / '05_decode_analysis.png'}")


# ── Entry point ───────────────────────────────────────────────────────────────

def generate_all_charts(csv_path: Path, out_dir: Path):
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
    chart_latency_distribution(rows, out_dir)
    chart_decode_share(agg, out_dir)

    print(f"\n  All charts saved to {out_dir}")


if __name__ == "__main__":
    import sys
    csv_path  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/benchmark_results.csv")
    out_dir   = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("charts")
    generate_all_charts(csv_path, out_dir)
