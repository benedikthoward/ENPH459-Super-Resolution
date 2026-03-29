"""
Re-generate poster figures from existing results.json with outlier rejection.
Removes trials where σ is more than 2× the median σ for that position.

    GENICAM_GENTL64_PATH=/opt/VimbaX_2026-1/cti uv run python -m imaging.calibration.replot_stability
"""

import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

from imaging import XPRController

# ── Point to existing data ──────────────────────────────────────────────────
DATA_DIR = Path("stability_data/20260329_010247")
FIGURES_DIR = Path.home() / "figures" / "runs" / "rolling_stability_pruned"
OUTLIER_FACTOR = 2.0           # reject trials with σ > factor × median(σ)
# ────────────────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]


def prune_outliers(sigma_all: list[float]) -> tuple[list[float], list[int]]:
    arr = np.array(sigma_all)
    med = np.median(arr)
    threshold = OUTLIER_FACTOR * med
    kept = []
    removed = []
    for i, s in enumerate(sigma_all):
        if s > threshold:
            removed.append(i)
        else:
            kept.append(s)
    return kept, removed


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "results.json") as f:
        results = json.load(f)

    params = results["params"]
    tilt_angle = params["tilt_angle"]
    n_trials = params["n_trials"]
    n_frames = params["num_frames"]
    angles = XPRController.get_xpr_angles(tilt_angle)

    print(f"Loaded {n_trials} trials × 4 positions × {n_frames} frames")
    print(f"Outlier threshold: {OUTLIER_FACTOR} × median(σ)\n")

    pruned = {}
    for p in range(4):
        key = f"pos{p}"
        sigma_all = results["aggregate"][key]["sigma_all"]
        kept, removed = prune_outliers(sigma_all)
        pruned[p] = {"kept": kept, "removed": removed, "original": sigma_all}
        if removed:
            print(f"  pos{p}: removed trials {removed} "
                  f"(σ = {[f'{sigma_all[i]:.3f}' for i in removed]}, "
                  f"median = {np.median(sigma_all):.3f}, "
                  f"threshold = {OUTLIER_FACTOR * np.median(sigma_all):.3f})")
        else:
            print(f"  pos{p}: no outliers (median σ = {np.median(sigma_all):.3f})")

    print()

    labels = [f"Position {p}\n({angles[p,0]:+.3f}, {angles[p,1]:+.3f})" for p in range(4)]
    x = np.arange(4)
    means_raw = [np.mean(pruned[p]["original"]) for p in range(4)]
    means = [np.mean(pruned[p]["kept"]) for p in range(4)]
    errs = [np.std(pruned[p]["kept"]) for p in range(4)]
    n_kept = [len(pruned[p]["kept"]) for p in range(4)]
    means_arr = np.array(means)
    errs_arr = np.array(errs)

    # ── Bar chart (pruned) ──
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(x, means, yerr=errs, capsize=6, color=COLORS, edgecolor="white",
                  linewidth=1.5, error_kw={"linewidth": 1.5, "capthick": 1.5})
    for i, (bar, m, e) in enumerate(zip(bars, means, errs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + e + 0.005,
                f"{m:.3f} ± {e:.3f} px\n({n_trials}/{n_trials} trials)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Edge Position Jitter σ (pixels)")
    ax.set_title(f"Beam-Shift Position Stability — Rolling Shutter (Outliers Removed)\n"
                 f"{n_trials} trials × {n_frames} frames, "
                 f"{params['exposure_us']/1000:.0f} ms exposure, 41 FPS")
    ax.set_ylim(0, max([m + e for m, e in zip(means, errs)]) * 1.6)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "bar_pruned.png"))
    print(f"Bar chart → bar_pruned.png")
    plt.close(fig)

    # ── Line plot (pruned) ──
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(x, means_arr, yerr=errs_arr, fmt="o-", color="#2196F3", markersize=10,
                linewidth=2.5, capsize=8, capthick=2, elinewidth=2,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
    ax.fill_between(x, means_arr - errs_arr, means_arr + errs_arr,
                    alpha=0.15, color="#2196F3")
    for i in range(4):
        ax.annotate(f"{means[i]:.3f} ± {errs[i]:.3f} px",
                    (x[i], means[i] + errs[i]),
                    textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Edge Position Jitter σ (pixels)")
    ax.set_title(f"Beam-Shift Position Stability — Rolling Shutter (Outliers Removed)\n"
                 f"{n_trials} trials × {n_frames} frames, "
                 f"{params['exposure_us']/1000:.0f} ms exposure, 41 FPS")
    ax.set_ylim(0, max(means_arr + errs_arr) * 1.6)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "line_pruned.png"))
    print(f"Line plot → line_pruned.png")
    plt.close(fig)

    # ── Before/after comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    errs_raw = [np.std(pruned[p]["original"]) for p in range(4)]
    bars = ax.bar(x, means_raw, yerr=errs_raw, capsize=5, color=COLORS, alpha=0.5,
                  edgecolor="white", linewidth=1.5, error_kw={"linewidth": 1.5})
    for i, (bar, m, e) in enumerate(zip(bars, means_raw, errs_raw)):
        ax.text(bar.get_x() + bar.get_width() / 2, m + e + 0.01,
                f"{m:.3f}±{e:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Pos {p}" for p in range(4)])
    ax.set_ylabel("Mean σ (px)")
    ax.set_title("Before Outlier Removal")
    ax.grid(True, alpha=0.2, axis="y")

    ax = axes[1]
    bars = ax.bar(x, means, yerr=errs, capsize=5, color=COLORS, edgecolor="white",
                  linewidth=1.5, error_kw={"linewidth": 1.5})
    for i, (bar, m, e) in enumerate(zip(bars, means, errs)):
        ax.text(bar.get_x() + bar.get_width() / 2, m + e + 0.01,
                f"{m:.3f}±{e:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Pos {p}" for p in range(4)])
    ax.set_ylabel("Mean σ (px)")
    ax.set_title("After Outlier Removal")
    ax.grid(True, alpha=0.2, axis="y")

    ymax = max(max([m + e for m, e in zip(means_raw, errs_raw)]),
               max([m + e for m, e in zip(means, errs)])) * 1.4
    axes[0].set_ylim(0, ymax)
    axes[1].set_ylim(0, ymax)
    fig.suptitle("Effect of Outlier Rejection (mechanical disturbances removed)", fontsize=14)
    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "before_after_comparison.png"))
    print(f"Comparison → before_after_comparison.png")
    plt.close(fig)

    # ── Histograms (from per-frame CSV, excluding outlier trials) ──
    import csv as csv_mod
    # load per-frame data
    per_frame = {p: [] for p in range(4)}
    with open(DATA_DIR / "edges.csv") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            trial = int(row["trial"])
            pos = int(row["position"])
            edge = float(row["edge_x"])
            if trial not in pruned[pos]["removed"]:
                per_frame[pos].append(edge)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for p in range(4):
        ax = axes[p // 2][p % 2]
        edges_arr = np.array(per_frame[p])
        edges_c = edges_arr - edges_arr.mean()
        sigma = edges_c.std()
        total = len(edges_c)

        ax.hist(edges_c, bins=80, alpha=0.75, color=COLORS[p],
                edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=1, alpha=0.4)
        ax.axvline(sigma, color="red", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.axvline(-sigma, color="red", linewidth=1.2, linestyle="--", alpha=0.7,
                   label=f"±σ = ±{sigma:.3f} px")
        ax.set_title(f"Position {p} ({angles[p,0]:+.3f}, {angles[p,1]:+.3f}) — "
                     f"σ = {sigma:.3f} px ({total} frames)")
        ax.set_xlabel("Δx (pixels)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.15)

    fig.suptitle(f"Edge Position Distribution — {n_trials} Trials Combined (Outliers Removed)",
                 fontsize=15)
    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "histograms_pruned.png"))
    print(f"Histograms → histograms_pruned.png")
    plt.close(fig)

    # save pruned JSON
    pruned_results = {
        "params": params,
        "outlier_factor": OUTLIER_FACTOR,
        "pruned_aggregate": {},
    }
    for p in range(4):
        key = f"pos{p}"
        pruned_results["pruned_aggregate"][key] = {
            "angle_x": float(angles[p, 0]),
            "angle_y": float(angles[p, 1]),
            "sigma_mean": float(np.mean(pruned[p]["kept"])),
            "sigma_std": float(np.std(pruned[p]["kept"])),
            "sigma_kept": pruned[p]["kept"],
            "trials_removed": pruned[p]["removed"],
            "n_kept": len(pruned[p]["kept"]),
        }
    (FIGURES_DIR / "results_pruned.json").write_text(json.dumps(pruned_results, indent=2))
    print(f"JSON → results_pruned.json")

    # copy to desktop
    desktop = Path.home() / "Desktop"
    for f in FIGURES_DIR.glob("*.png"):
        shutil.copy2(f, desktop / f.name)
    print(f"\nAll figures copied to ~/Desktop/")
    print("Done.")


if __name__ == "__main__":
    main()
