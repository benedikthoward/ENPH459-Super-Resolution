"""
Rolling shutter stability: stream frames from Allied Vision camera
at max FPS, track knife-edge jitter at each of the 4 XPR positions.

Runs N_TRIALS independent trials, then generates poster-quality figures
with error bars across trials.

    GENICAM_GENTL64_PATH=/opt/VimbaX_2026-1/cti uv run python -m imaging.calibration.rolling_stability
"""

import csv
import json
import shutil
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.allied_vision_camera import AlliedCamera
from api import XPRController

# ── Parameters ──────────────────────────────────────────────────────────────
TILT_ANGLE = 0.14391           # rad
NUM_FRAMES = 1000              # frames per position per trial
N_TRIALS = 10                  # independent repetitions
EXPOSURE_US = 20000            # µs
ROI_HEIGHT = 512
ROI_WIDTH = 5496
SAVE_RAW = False               # skip raw image saving for speed
FIGURES_DIR = Path.home() / "figures" / "runs"
OUTPUT_DIR = Path("./stability_data")
# ────────────────────────────────────────────────────────────────────────────

# poster style
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


def find_edge_position(img: np.ndarray) -> float:
    if img.ndim == 3:
        img = img[:, :, 0]
    profile = img.astype(np.float64).mean(axis=0)
    mid = (profile.min() + profile.max()) / 2.0
    for i in range(len(profile) - 1):
        if (profile[i] <= mid < profile[i + 1]) or (profile[i] >= mid > profile[i + 1]):
            t = (mid - profile[i]) / (profile[i + 1] - profile[i])
            return i + t
    return float(np.argmin(np.abs(profile - mid)))


def run_single_trial(cam, xpr, angles, trial_idx):
    """Run one trial at all 4 positions. Returns dict[pos] -> {edges, timestamps, fps}."""
    data = {}
    for p in range(4):
        xpr.set_angles(angles[p, 0], angles[p, 1])
        time.sleep(0.02)

        print(f"    pos{p}: streaming...", end=" ", flush=True)
        images, timestamps = cam.stream_burst(NUM_FRAMES)
        if len(images) == 0:
            print("FAILED (0 frames). Retrying...", end=" ", flush=True)
            time.sleep(1)
            images, timestamps = cam.stream_burst(NUM_FRAMES)
        if len(images) == 0:
            raise RuntimeError(f"Camera returned 0 frames at pos{p}")
        fps = len(images) / (timestamps[-1] / 1000)
        edges = [find_edge_position(img) for img in images]
        del images  # free ~2.7 GB per position
        std = np.std(edges)
        print(f"{fps:.1f} FPS, σ={std:.4f} px")
        data[p] = {"edges": edges, "timestamps": timestamps, "fps": fps}

    xpr.set_home()
    time.sleep(0.05)
    return data


def plot_poster_bar(all_trials, angles, fig_path):
    """Bar chart: mean σ per position with std-of-σ error bars across trials."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    means = []
    errs = []
    for p in range(4):
        sigmas = [np.std(all_trials[t][p]["edges"]) for t in range(N_TRIALS)]
        means.append(np.mean(sigmas))
        errs.append(np.std(sigmas))

    x = np.arange(4)
    bars = ax.bar(x, means, yerr=errs, capsize=6, color=COLORS, edgecolor="white",
                  linewidth=1.5, error_kw={"linewidth": 1.5, "capthick": 1.5})

    for i, (bar, m, e) in enumerate(zip(bars, means, errs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + e + 0.003,
                f"{m:.3f} ± {e:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    labels = [f"Position {p}\n({angles[p,0]:+.3f}, {angles[p,1]:+.3f})" for p in range(4)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Edge Position Jitter σ (pixels)")
    ax.set_title(f"Beam-Shift Position Stability — Rolling Shutter\n"
                 f"{N_TRIALS} trials × {NUM_FRAMES} frames, {EXPOSURE_US/1000:.0f} ms exposure")
    ax.set_ylim(0, max(means) * 1.5)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    fig.savefig(str(fig_path))
    print(f"  Bar chart → {fig_path}")
    plt.close(fig)


def plot_poster_line(all_trials, angles, fig_path):
    """Line plot: σ per position with error bars, connected by lines."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    x = np.arange(4)
    means = []
    errs = []
    for p in range(4):
        sigmas = [np.std(all_trials[t][p]["edges"]) for t in range(N_TRIALS)]
        means.append(np.mean(sigmas))
        errs.append(np.std(sigmas))

    means = np.array(means)
    errs = np.array(errs)

    ax.errorbar(x, means, yerr=errs, fmt="o-", color="#2196F3", markersize=10,
                linewidth=2, capsize=8, capthick=2, elinewidth=2,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)

    # shade the error region
    ax.fill_between(x, means - errs, means + errs, alpha=0.15, color="#2196F3")

    # annotate each point
    for i in range(4):
        ax.annotate(f"{means[i]:.3f} ± {errs[i]:.3f} px",
                    (x[i], means[i] + errs[i]),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=10, fontweight="bold")

    labels = [f"Pos {p}\n({angles[p,0]:+.3f}, {angles[p,1]:+.3f})" for p in range(4)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Edge Position Jitter σ (pixels)")
    ax.set_title(f"Beam-Shift Position Stability — Rolling Shutter\n"
                 f"{N_TRIALS} trials × {NUM_FRAMES} frames, {EXPOSURE_US/1000:.0f} ms exposure")
    ax.set_ylim(0, max(means + errs) * 1.5)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    fig.savefig(str(fig_path))
    print(f"  Line plot → {fig_path}")
    plt.close(fig)


def plot_poster_timeseries(all_trials, angles, fig_path):
    """Overlay all trials' time series per position (2×2 grid)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for p in range(4):
        ax = axes[p // 2][p % 2]
        for t in range(N_TRIALS):
            edges = np.array(all_trials[t][p]["edges"])
            timestamps = np.array(all_trials[t][p]["timestamps"])
            edges_c = edges - edges.mean()
            ax.plot(timestamps / 1000, edges_c, linewidth=0.3, alpha=0.5, color=COLORS[p])

        # compute aggregate stats
        all_sigmas = [np.std(all_trials[t][p]["edges"]) for t in range(N_TRIALS)]
        mean_sigma = np.mean(all_sigmas)

        ax.axhline(0, color="k", alpha=0.3, linewidth=0.8)
        ax.axhline(mean_sigma, color="red", alpha=0.5, linewidth=1, linestyle="--")
        ax.axhline(-mean_sigma, color="red", alpha=0.5, linewidth=1, linestyle="--")
        ax.set_title(f"Position {p} ({angles[p,0]:+.3f}, {angles[p,1]:+.3f}) — "
                     f"mean σ = {mean_sigma:.3f} px")
        ax.set_ylabel("Δx (pixels)")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.15)

    fig.suptitle(f"Edge Position Time Series — {N_TRIALS} Trials Overlaid", fontsize=15)
    plt.tight_layout()
    fig.savefig(str(fig_path))
    print(f"  Time series → {fig_path}")
    plt.close(fig)


def plot_poster_histograms(all_trials, angles, fig_path):
    """Stacked histograms per position (2×2 grid), all trials combined."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for p in range(4):
        ax = axes[p // 2][p % 2]
        all_edges = []
        for t in range(N_TRIALS):
            edges = np.array(all_trials[t][p]["edges"])
            all_edges.append(edges - edges.mean())
        all_edges = np.concatenate(all_edges)

        ax.hist(all_edges, bins=80, alpha=0.75, color=COLORS[p],
                edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=1, alpha=0.4)

        sigma = all_edges.std()
        ax.axvline(sigma, color="red", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.axvline(-sigma, color="red", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.set_title(f"Position {p} ({angles[p,0]:+.3f}, {angles[p,1]:+.3f}) — "
                     f"σ = {sigma:.3f} px ({N_TRIALS * NUM_FRAMES} samples)")
        ax.set_xlabel("Δx (pixels)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.15)

    fig.suptitle(f"Edge Position Distribution — All Trials Combined", fontsize=15)
    plt.tight_layout()
    fig.savefig(str(fig_path))
    print(f"  Histograms → {fig_path}")
    plt.close(fig)


def run():
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / run_ts
    out.mkdir(parents=True, exist_ok=True)

    fig_dir = FIGURES_DIR / f"rolling_stability_{run_ts}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    angles = XPRController.get_xpr_angles(TILT_ANGLE)

    print(f"Rolling Shutter Stability — {N_TRIALS} trials × 4 positions × {NUM_FRAMES} frames")
    print(f"Output: {out}")
    print(f"Figures: {fig_dir}")
    print()

    all_trials = []  # list of dicts, one per trial
    all_csv_rows = []

    with AlliedCamera(exposure_us=EXPOSURE_US) as cam, XPRController() as xpr:
        cam.width = ROI_WIDTH
        cam.height = ROI_HEIGHT

        print(f"Camera: {cam.model}, {cam.width}x{cam.height}")
        print(f"Exposure: {cam.exposure} µs, Max FPS: {cam.max_fps:.1f}")
        print()

        params = {
            "camera": cam.model,
            "roi": {"width": ROI_WIDTH, "height": ROI_HEIGHT},
            "exposure_us": EXPOSURE_US,
            "tilt_angle": TILT_ANGLE,
            "num_frames": NUM_FRAMES,
            "n_trials": N_TRIALS,
            "timestamp": run_ts,
        }
        (out / "params.json").write_text(json.dumps(params, indent=2))

        for trial in range(N_TRIALS):
            print(f"  Trial {trial + 1}/{N_TRIALS}:")
            data = run_single_trial(cam, xpr, angles, trial)
            all_trials.append(data)

            for p in range(4):
                for i, (edge, ts) in enumerate(zip(data[p]["edges"], data[p]["timestamps"])):
                    all_csv_rows.append([trial, p, i, ts, edge])

            print()

    # save CSV
    csv_path = out / "edges.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "position", "frame", "time_ms", "edge_x"])
        writer.writerows(all_csv_rows)
    print(f"CSV → {csv_path}")

    # save per-trial and aggregate JSON
    trial_summaries = []
    for t in range(N_TRIALS):
        trial_data = {}
        for p in range(4):
            e = np.array(all_trials[t][p]["edges"])
            trial_data[f"pos{p}"] = {
                "angle_x": float(angles[p, 0]),
                "angle_y": float(angles[p, 1]),
                "edge_std_px": float(e.std()),
                "edge_range_px": float(e.max() - e.min()),
                "edge_mean_px": float(e.mean()),
                "fps": all_trials[t][p]["fps"],
            }
        trial_summaries.append(trial_data)

    aggregate = {}
    for p in range(4):
        sigmas = [np.std(all_trials[t][p]["edges"]) for t in range(N_TRIALS)]
        means = [np.mean(all_trials[t][p]["edges"]) for t in range(N_TRIALS)]
        aggregate[f"pos{p}"] = {
            "angle_x": float(angles[p, 0]),
            "angle_y": float(angles[p, 1]),
            "sigma_mean": float(np.mean(sigmas)),
            "sigma_std": float(np.std(sigmas)),
            "sigma_all": [float(s) for s in sigmas],
            "edge_mean_all": [float(m) for m in means],
        }

    results = {
        "params": params,
        "aggregate": aggregate,
        "per_trial": trial_summaries,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"JSON → {out / 'results.json'}")

    # generate poster figures
    print("\nGenerating figures:")
    plot_poster_bar(all_trials, angles, fig_dir / "stability_bar.png")
    plot_poster_line(all_trials, angles, fig_dir / "stability_line.png")
    plot_poster_timeseries(all_trials, angles, fig_dir / "stability_timeseries.png")
    plot_poster_histograms(all_trials, angles, fig_dir / "stability_histograms.png")

    # copy to data dir too
    for f in fig_dir.glob("*.png"):
        shutil.copy2(f, out / f.name)

    # write README
    (fig_dir / "README.txt").write_text(
        f"Rolling Shutter Stability Experiment\n"
        f"{'=' * 40}\n"
        f"Camera: Allied Vision 1800 U-2050c (rolling shutter, IMX183)\n"
        f"ROI: {ROI_WIDTH}x{ROI_HEIGHT}, Exposure: {EXPOSURE_US} µs\n"
        f"Tilt angle: {TILT_ANGLE} rad\n"
        f"Trials: {N_TRIALS}, Frames per position per trial: {NUM_FRAMES}\n"
        f"Total frames: {N_TRIALS * 4 * NUM_FRAMES}\n"
        f"Timestamp: {run_ts}\n\n"
        f"Figures:\n"
        f"  stability_bar.png         — Bar chart: mean σ ± std across trials\n"
        f"  stability_line.png        — Line plot: same data with error band\n"
        f"  stability_timeseries.png  — All trials overlaid per position\n"
        f"  stability_histograms.png  — Combined histograms per position\n"
    )

    # copy to desktop
    desktop = Path.home() / "Desktop"
    for f in fig_dir.glob("*.png"):
        shutil.copy2(f, desktop / f.name)
    print(f"\nFigures also copied to ~/Desktop/")

    print("Done.")


if __name__ == "__main__":
    run()
