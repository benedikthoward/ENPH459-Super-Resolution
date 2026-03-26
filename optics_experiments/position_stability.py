"""
Position stability: burst-capture at each of the 4 XPR positions and
track knife-edge position jitter over time.

Uses short exposure for max FPS. Captures first, then processes.

    uv run python -m optics_experiments.position_stability
"""

import csv
import json
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from imaging import XPRController, DahengCamera

# ── Parameters ──────────────────────────────────────────────────────────────
TILT_ANGLE = 0.14391           # rad (Daheng mono default)
NUM_FRAMES = 500               # frames per position (burst, no sleep)
EXPOSURES = [10000, 5000]      # µs — run experiment at each exposure
GAIN = 0                       # dB
OUTPUT_DIR = Path("./stability_data")
# ────────────────────────────────────────────────────────────────────────────


def find_edge_position(img: np.ndarray) -> float:
    """Sub-pixel x-position of a vertical knife edge (dark left, bright right).
    Averages all rows into a 1D profile, finds the 50% crossing point."""
    profile = img.astype(np.float64).mean(axis=0)
    mid = (profile.min() + profile.max()) / 2.0
    for i in range(len(profile) - 1):
        if (profile[i] <= mid < profile[i + 1]) or (profile[i] >= mid > profile[i + 1]):
            t = (mid - profile[i]) / (profile[i + 1] - profile[i])
            return i + t
    return float(np.argmin(np.abs(profile - mid)))


def plot_results(all_data: dict, out: Path):
    """Plot results for all exposures combined."""
    n_exp = len(all_data)

    # ── Figure 1: edge position time series ──
    fig1, axes1 = plt.subplots(4, n_exp, figsize=(7 * n_exp, 12), squeeze=False)
    for ei, (exp_us, data) in enumerate(all_data.items()):
        for p in range(4):
            ax = axes1[p][ei]
            t = np.array(data[p]["time_ms"])
            t = t - t[0]
            edge = np.array(data[p]["edge_x"])
            edge_centered = edge - edge.mean()
            ax.plot(t, edge_centered, linewidth=0.5)
            ax.set_ylabel(f"pos{p} Δx (px)")
            ax.set_title(f"pos{p} @ {exp_us}µs — σ={edge.std():.4f} px" if p == 0 or ei > 0
                         else f"pos{p} — σ={edge.std():.4f} px")
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="k", alpha=0.2)
        axes1[-1][ei].set_xlabel("Time (ms)")
    fig1.suptitle("Edge position drift over time", fontsize=14)
    plt.tight_layout()
    fig1.savefig(str(out / "timeseries.png"), dpi=150)

    # ── Figure 2: histogram of edge positions ──
    fig2, axes2 = plt.subplots(4, n_exp, figsize=(7 * n_exp, 10), squeeze=False)
    for ei, (exp_us, data) in enumerate(all_data.items()):
        for p in range(4):
            ax = axes2[p][ei]
            edge = np.array(data[p]["edge_x"])
            edge_centered = edge - edge.mean()
            ax.hist(edge_centered, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.axvline(0, color="r", linestyle="--", alpha=0.5)
            ax.set_title(f"pos{p} @ {exp_us}µs — σ={edge.std():.4f} px")
            ax.set_xlabel("Δx (px)")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
    fig2.suptitle("Edge position distribution", fontsize=14)
    plt.tight_layout()
    fig2.savefig(str(out / "histograms.png"), dpi=150)

    # ── Figure 3: summary bar chart ──
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(4)
    bar_w = 0.8 / n_exp

    for ei, (exp_us, data) in enumerate(all_data.items()):
        stds = [np.array(data[p]["edge_x"]).std() for p in range(4)]
        offset = (ei - n_exp / 2 + 0.5) * bar_w
        bars = ax3.bar(x_pos + offset, stds, bar_w, label=f"{exp_us} µs")
        for i, (bar, s) in enumerate(zip(bars, stds)):
            fps = data[i]["fps"]
            ax3.annotate(f"{s:.4f} px\n{fps:.0f} FPS",
                         (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         textcoords="offset points", xytext=(0, 5),
                         ha="center", fontsize=8)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"pos {p}" for p in range(4)])
    ax3.set_ylabel("Edge position σ (px)")
    ax3.set_title("Position stability summary")
    ax3.legend(title="Exposure")
    ax3.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig3.savefig(str(out / "summary.png"), dpi=150)

    plt.show()
    print(f"Figures saved to {out}")


def run():
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / run_ts
    out.mkdir(parents=True, exist_ok=True)

    with XPRController() as xpr, DahengCamera() as cam:
        cam.gain = GAIN
        angles = XPRController.get_xpr_angles(TILT_ANGLE)

        params = {
            "tilt_angle": TILT_ANGLE,
            "num_frames": NUM_FRAMES,
            "exposures_us": EXPOSURES,
            "gain_db": GAIN,
            "camera": {"width": cam.width, "height": cam.height, "is_color": cam.is_color},
            "timestamp": run_ts,
        }
        (out / "params.json").write_text(json.dumps(params, indent=2))

        print(f"Saving to {out}")
        print(f"Camera: {cam.width}x{cam.height}")
        print(f"Tilt angle: {TILT_ANGLE} rad, {NUM_FRAMES} frames per position")
        print(f"Exposures to test: {EXPOSURES} µs")
        print()

        all_data = {}  # all_data[exp_us][pos] = {edge_x, time_ms, fps}
        csv_rows = []

        for exp_us in EXPOSURES:
            cam.exposure = exp_us
            print(f"=== Exposure: {exp_us} µs, Gain: {GAIN} dB ===")

            data = {}
            exp_dir = out / f"exp{exp_us}us"
            exp_dir.mkdir()

            for p in range(4):
                pos_dir = exp_dir / f"pos{p}"
                pos_dir.mkdir()

                xpr.set_angles(angles[p, 0], angles[p, 1])
                time.sleep(0.02)  # 20ms settling

                # burst capture
                print(f"  pos{p}: capturing {NUM_FRAMES} frames...", end=" ", flush=True)
                images = []
                timestamps = []
                t0 = time.perf_counter()
                for f in range(NUM_FRAMES):
                    img = cam.capture_raw()
                    timestamps.append((time.perf_counter() - t0) * 1000)
                    images.append(img)
                elapsed = timestamps[-1]
                fps = NUM_FRAMES / (elapsed / 1000)
                print(f"{fps:.1f} FPS ({elapsed:.0f} ms)")

                # save images + find edge
                print(f"         processing...", end=" ", flush=True)
                edges = []
                for f, img in enumerate(images):
                    cv2.imwrite(str(pos_dir / f"frame_{f:03d}.png"), img)
                    edge_x = find_edge_position(img)
                    edges.append(edge_x)
                    csv_rows.append([exp_us, p, f, edge_x, timestamps[f]])

                data[p] = {"edge_x": edges, "time_ms": timestamps, "fps": fps}
                print(f"σ={np.std(edges):.4f} px, mean={np.mean(edges):.2f} px")

            xpr.set_home()
            all_data[exp_us] = data
            print()

    # save CSV
    csv_path = out / "edges.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["exposure_us", "position", "frame", "edge_x", "time_ms"])
        writer.writerows(csv_rows)
    print(f"Data saved to {csv_path}")

    # save JSON summary
    summary = {}
    for exp_us, data in all_data.items():
        for p in range(4):
            edge = np.array(data[p]["edge_x"])
            key = f"exp{exp_us}_pos{p}"
            summary[key] = {
                "exposure_us": exp_us,
                "position": p,
                "edge_mean": float(edge.mean()),
                "edge_std": float(edge.std()),
                "edge_range": float(edge.max() - edge.min()),
                "fps": data[p]["fps"],
                "num_frames": NUM_FRAMES,
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_results(all_data, out)
    print("Done.")


if __name__ == "__main__":
    run()
