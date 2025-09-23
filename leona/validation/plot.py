#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Latency plots from summary_inference.csv (with hardware split).

- Splits figures by hardware (hw column: e.g., rpi4 / jetson)
- Distinguishes "e2e" (full runs) vs "e2e_simple" (single-sample)
- Uses "CNN" and "RNN" labels
- Prettier, rounded visuals (pure Matplotlib)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

# =========================
# CONFIG
# =========================
CSV_PATH = "summary2.csv"
OUTDIR   = "figs"

# Consistent model colors/markers everywhere
COLOR_MAP  = {"CNN": "tab:blue", "RNN": "tab:orange"}
MARKERS    = {"CNN": "o", "RNN": "s"}

# =========================
# Style (soft + rounded)
# =========================
def set_pretty_style():
    plt.rcParams.update({
        # fonts & sizes
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,

        # figure & axes facecolors
        "figure.facecolor": "#ffffff",
        "axes.facecolor":  "#fcfdff",

        # grid
        "axes.grid": True,
        "grid.color": "#aab4c0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.5,

        # lines & patches — rounded ends/joins
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "patch.edgecolor": "#333333",
        "patch.linewidth": 0.8,

        # spines
        "axes.spines.top": False,
        "axes.spines.right": False,

        # savefig
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "savefig.transparent": False,
    })

def rounded_legend(**kwargs):
    return dict(frameon=True, framealpha=0.9, fancybox=True, borderpad=0.3, **kwargs)

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def savefig(path_base: str):
    plt.tight_layout()
    plt.savefig(path_base + ".png", dpi=200)
    plt.savefig(path_base + ".svg")
    plt.close()

# =========================
# Helpers
# =========================
def normalize_model_name(name: str) -> str:
    n = (name or "").strip().lower()
    if "cnn" in n: return "CNN"
    if "rnn" in n: return "RNN"
    return name

def normalize_run_type(rt: str) -> str:
    rt = (rt or "").strip().lower()
    if rt == "e2e": return "e2e"
    if "simple" in rt: return "e2e_simple"
    return rt

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"model_name","inference_type","total_samples","total_wall_ms","avg_ms","p50_ms","p99_ms","hw"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"CSV missing columns: {sorted(miss)}")

    df["model"]    = df["model_name"].map(normalize_model_name)
    df["run_type"] = df["inference_type"].map(normalize_run_type)
    df["hw"]       = df["hw"].astype(str).str.lower()
    df["throughput_sps"] = df["total_samples"] / (df["total_wall_ms"] / 1000.0)
    return df

# =========================
# Plotters (per hardware)
# =========================
def plot_boxplot(df_hw: pd.DataFrame, run_type: str, hw: str):
    sub = df_hw[df_hw["run_type"] == run_type]
    if sub.empty: return
    plt.figure(figsize=(7.2,5.2))

    groups, labels = [], []
    for model, g in sub.groupby("model"):
        groups.append(g["avg_ms"].values)
        labels.append(model)

    box = plt.boxplot(
        groups, labels=labels, showmeans=True, patch_artist=True,
        medianprops=dict(color="#1f2937", linewidth=2),
        boxprops=dict(linewidth=1.0),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        meanprops=dict(marker="D", markerfacecolor="#ffffff", markeredgecolor="#1f2937", markersize=6)
    )
    # fill boxes w/ soft colors
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(COLOR_MAP.get(label, "#cccccc"))
        patch.set_alpha(0.25)

    plt.ylabel("Average latency per sample (ms)")
    plt.title(f"{run_type.upper()} runs on {hw}: CNN vs RNN latency")
    plt.legend(**rounded_legend(loc="best", title=None)).remove()  # no legend needed for boxplot
    savefig(os.path.join(OUTDIR, f"boxplot_{run_type}_avg_ms_{hw}"))

def plot_hist(df_hw: pd.DataFrame, run_type: str, hw: str):
    sub = df_hw[df_hw["run_type"] == run_type]
    if sub.empty: return
    plt.figure(figsize=(7.2,5.2))

    bins = np.linspace(sub["avg_ms"].min()*0.9, sub["avg_ms"].max()*1.1, 15)
    for model, g in sub.groupby("model"):
        n, bins_, patches = plt.hist(
            g["avg_ms"].values, bins=bins, alpha=0.4, label=model,
            color=COLOR_MAP.get(model, None), edgecolor="none"
        )
        # soften bar edges
        for p in patches:
            p.set_linewidth(0)

    plt.xlabel("Average latency per sample (ms)")
    plt.ylabel("Number of runs")
    plt.title(f"{run_type.upper()} runs on {hw}: latency distribution")
    plt.legend(**rounded_legend(loc="best", title="Model"))
    savefig(os.path.join(OUTDIR, f"hist_{run_type}_avg_ms_{hw}"))

def plot_scatter_p50_vs_p99(df_hw: pd.DataFrame, run_type: str, hw: str):
    sub = df_hw[df_hw["run_type"] == run_type]
    if sub.empty: return
    plt.figure(figsize=(7.2,5.2))
    for model, g in sub.groupby("model"):
        plt.scatter(
            g["p50_ms"], g["p99_ms"],
            label=model, alpha=0.9,
            color=COLOR_MAP.get(model, None),
            marker=MARKERS.get(model, "o"),
            s=70, linewidths=0.6, edgecolors="#1f2937"
        )
    plt.xlabel("p50 latency (ms) — typical")
    plt.ylabel("p99 latency (ms) — tail")
    plt.title(f"{run_type.upper()} runs on {hw}: Typical vs Tail (p50 vs p99)")
    plt.legend(**rounded_legend(loc="best", title="Model"))
    savefig(os.path.join(OUTDIR, f"scatter_{run_type}_p50_vs_p99_{hw}"))

def plot_percentile_curves_from_runs(df_hw: pd.DataFrame, hw: str):
    """Percentile curves across runs using per-run avg_ms (only e2e)."""
    sub = df_hw[df_hw["run_type"] == "e2e"].copy()
    if sub.empty: return

    plt.figure(figsize=(9.2,6.0))
    ax = plt.gca()

    ymax = 0  # track max y for padding

    for model, g in sub.groupby("model"):
        lat = np.sort(g["avg_ms"].values)
        pct = np.linspace(0, 100, len(lat))

        (line,) = ax.plot(
            pct, lat,
            label=model,
            color=COLOR_MAP.get(model, None),
            linewidth=2.4,
            marker=MARKERS.get(model, "o"),
            markersize=4.5,
            alpha=0.95
        )

        for p in [50, 90, 95, 99]:
            val = np.percentile(lat, p)
            ax.scatter([p], [val],
                       color=COLOR_MAP.get(model, None),
                       marker="x", s=90, linewidths=2.2, zorder=5)

            if p == 50:
                xytext = (p, val + 0.005); va, ha = "bottom", "center"
            elif p == 90:
                xytext = (p - 7, val - 0.006); va, ha = "top", "right"
            elif p == 95:
                xytext = (p, val + 0.005); va, ha = "bottom", "center"
            else:  # 99
                xytext = (p, val - 0.006); va, ha = "top", "center"

            ax.annotate(
                f"{p}th = {val:.3f} ms",
                xy=(p, val), xytext=xytext, textcoords="data",
                ha=ha, va=va, fontsize=10,
                color=COLOR_MAP.get(model,"black"),
                arrowprops=dict(arrowstyle="->", color=COLOR_MAP.get(model,None), shrinkA=4, shrinkB=4),
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5)
            )

            ymax = max(ymax, val)

        # shaded band
        p50 = np.percentile(lat, 50)
        p99 = np.percentile(lat, 99)
        ax.fill_between(pct, p50, p99, color=COLOR_MAP.get(model, None), alpha=0.10)

    ax.set_xlabel("Percentile across runs (%)")
    ax.set_ylabel("Latency per sample (ms)")
    ax.set_title(f"Latency Percentiles by Model (avg_ms across runs) — {hw}")
    ax.legend(title="Model", frameon=True, framealpha=0.9, fancybox=True)

    # 🔑 Add 15% headroom on y-axis
    ymin, _ = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.15)

    savefig(os.path.join(OUTDIR, f"percentiles_avg_ms_by_model_{hw}"))

def plot_percentiles_4up(df: pd.DataFrame, run_type: str = "e2e"):
    """
    One plot with 4 lines: CNN-rpi4, CNN-jetson, RNN-rpi4, RNN-jetson.
    Uses per-run avg_ms to compute percentiles across runs for each (model, hw).
    """
    sub = df[(df["run_type"] == run_type) & (df["hw"].isin(["rpi4", "jetson"]))].copy()
    if sub.empty:
        return

    plt.figure(figsize=(10,6))
    ax = plt.gca()

    # color by model; style by hardware
    color_map = {"CNN": "tab:blue", "RNN": "tab:orange"}
    ls_map    = {"rpi4": "-", "jetson": "--"}
    marker_map= {"rpi4": "o", "jetson": "v"}

    ymax = 0.0

    # ensure deterministic legend order
    for model in ["CNN", "RNN"]:
        for hw in ["rpi4", "jetson"]:
            g = sub[(sub["model"] == model) & (sub["hw"] == hw)]
            if g.empty:
                continue

            lat = np.sort(g["avg_ms"].values)
            pct = np.linspace(0, 100, len(lat))

            (line,) = ax.plot(
                pct, lat,
                label=f"{model} • {hw}",
                color=color_map.get(model),
                linestyle=ls_map.get(hw, "-"),
                marker=marker_map.get(hw, "o"),
                markersize=4.5,
                linewidth=2.4,
                alpha=0.95,
            )
            line.set_solid_capstyle("round")
            line.set_solid_joinstyle("round")

            # percentile markers + staggered labels
            for p in [50, 90, 95, 99]:
                val = np.percentile(lat, p)
                ax.scatter([p], [val],
                           color=color_map.get(model),
                           marker="x", s=90, linewidths=2.2, zorder=5)

                # 50/95 above; 99 below; 90 shifted left + below
                if p == 50:
                    xytext = (p, val + 0.005); va, ha = "bottom", "center"
                elif p == 90:
                    xytext = (p - 7, val - 0.006); va, ha = "top", "right"
                elif p == 95:
                    xytext = (p, val + 0.005); va, ha = "bottom", "center"
                else:  # 99
                    xytext = (p, val - 0.006); va, ha = "top", "center"

                ax.annotate(
                    f"{p}th = {val:.3f} ms",
                    xy=(p, val), xytext=xytext, textcoords="data",
                    ha=ha, va=va, fontsize=10, color=color_map.get(model),
                    arrowprops=dict(arrowstyle="->", color=color_map.get(model), shrinkA=4, shrinkB=4),
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5)
                )

                ymax = max(ymax, val)

            # shaded band p50..p99 per line (subtle)
            p50 = np.percentile(lat, 50)
            p99 = np.percentile(lat, 99)
            ax.fill_between(pct, p50, p99, color=color_map.get(model), alpha=0.08)

    ax.set_xlabel("Percentile across runs (%)")
    ax.set_ylabel("Latency per sample (ms)")
    ax.set_title(f"Latency Percentiles across HW ({run_type}) — CNN/RNN on rpi4 vs jetson")
    ax.legend(title="Model • HW", frameon=True, framealpha=0.9, fancybox=True)

    # headroom so labels don’t hit the title
    ymin, _ = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.18)

    savefig(os.path.join(OUTDIR, f"percentiles_4up_{run_type}"))

def plot_boxplot_4up(df: pd.DataFrame, run_type: str):
    sub = df[df["run_type"] == run_type]
    if sub.empty: return

    plt.figure(figsize=(8,6))
    groups, labels = [], []
    for model in ["CNN", "RNN"]:
        for hw in ["rpi4", "jetson"]:
            g = sub[(sub["model"] == model) & (sub["hw"] == hw)]
            if g.empty: continue
            groups.append(g["avg_ms"].values)
            labels.append(f"{model}-{hw}")

    bp = plt.boxplot(groups, labels=labels, showmeans=True,
                     patch_artist=True, notch=True, medianprops=dict(color="black"))

    # colorize boxes
    colors = ["tab:blue","tab:blue","tab:orange","tab:orange"]
    hatches = ["/","\\","/","\\"] * (len(bp["boxes"])//4 + 1)
    for patch, color, hatch in zip(bp["boxes"], colors, hatches):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
        patch.set_hatch(hatch)

    plt.ylabel("Average latency per sample (ms)")
    plt.title(f"{run_type.upper()} runs: CNN vs RNN latency on rpi4/jetson")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    savefig(os.path.join(OUTDIR, f"boxplot_4up_{run_type}"))


def plot_hist_4up(df: pd.DataFrame, run_type: str):
    sub = df[df["run_type"] == run_type]
    if sub.empty: return

    plt.figure(figsize=(8,6))
    bins = np.linspace(sub["avg_ms"].min()*0.9, sub["avg_ms"].max()*1.1, 20)
    styles = {("CNN","rpi4"):"tab:blue", ("CNN","jetson"):"deepskyblue",
              ("RNN","rpi4"):"tab:orange", ("RNN","jetson"):"goldenrod"}

    for (model, hw), g in sub.groupby(["model","hw"]):
        plt.hist(g["avg_ms"].values, bins=bins, alpha=0.5,
                 color=styles.get((model,hw),"gray"), label=f"{model}-{hw}")

    plt.xlabel("Average latency per sample (ms)")
    plt.ylabel("Number of runs")
    plt.title(f"{run_type.upper()} runs: latency distribution on rpi4/jetson")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    savefig(os.path.join(OUTDIR, f"hist_4up_{run_type}"))


def plot_scatter_4up(df: pd.DataFrame, run_type: str):
    sub = df[df["run_type"] == run_type]
    if sub.empty: return

    plt.figure(figsize=(8,6))
    styles = {("CNN","rpi4"):("tab:blue","o"),
              ("CNN","jetson"):("deepskyblue","^"),
              ("RNN","rpi4"):("tab:orange","s"),
              ("RNN","jetson"):("goldenrod","v")}

    for (model, hw), g in sub.groupby(["model","hw"]):
        c, m = styles.get((model,hw),("gray","x"))
        plt.scatter(g["p50_ms"], g["p99_ms"], label=f"{model}-{hw}",
                    alpha=0.8, color=c, marker=m, s=70, edgecolor="k")

    plt.xlabel("p50 latency (ms)")
    plt.ylabel("p99 latency (ms)")
    plt.title(f"{run_type.upper()} runs: Typical vs Tail Latency (p50 vs p99) on rpi4/jetson")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Model-HW")
    savefig(os.path.join(OUTDIR, f"scatter_4up_{run_type}"))

# =========================
# Main
# =========================
def main():
    set_pretty_style()
    ensure_outdir(OUTDIR)
    df = load_data(CSV_PATH)

    # Split by hardware and generate a full set for each
    for hw, df_hw in df.groupby("hw"):
        for run_type in ["e2e", "e2e_simple"]:
            plot_boxplot(df_hw, run_type, hw)
            plot_hist(df_hw, run_type, hw)
            plot_scatter_p50_vs_p99(df_hw, run_type, hw)
            plot_boxplot_4up(df, run_type)
            plot_hist_4up(df, run_type)
            plot_scatter_4up(df, run_type)
        plot_percentile_curves_from_runs(df_hw, hw)
        plot_percentiles_4up(df, run_type="e2e")
        plot_percentiles_4up(df, run_type="e2e_simple")

    print("\nDone. Figures saved in:", OUTDIR)

if __name__ == "__main__":
    main()
