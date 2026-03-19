"""
Position stability: burst-capture at each of the 4 XPR positions and
track pinhole centroid jitter over time.

Uses short exposure + gain for max FPS. Captures first, then processes.

    uv run python -m imaging.calibration.position_stability
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
EXPOSURE = 5000                # µs (5ms for high FPS)
GAIN = 10                      # dB (compensate short exposure)
ROI_SIZE = 100                 # px crop around pinhole for centroid
OUTPUT_DIR = Path("./stability_data")
# ────────────────────────────────────────────────────────────────────────────


def find_pinhole_center(img: np.ndarray, roi_size: int = ROI_SIZE) -> tuple[float, float]:
    """Sub-pixel centroid of bright pinhole, cropped to exclude far-field noise."""
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    mx, my = max_loc

    h, w = img.shape
    half = roi_size // 2
    x1, y1 = max(0, mx - half), max(0, my - half)
    x2, y2 = min(w, mx + half), min(h, my + half)
    roi = img[y1:y2, x1:x2].astype(np.float64)

    bg = np.median(roi)
    roi = np.clip(roi - bg, 0, None)
    roi[roi < roi.max() * 0.1] = 0

    M = cv2.moments(roi)
    if M["m00"] == 0:
        return float(mx), float(my)
    return M["m10"] / M["m00"] + x1, M["m01"] / M["m00"] + y1


def plot_results(data: dict, out: Path):
    angles = XPRController.get_xpr_angles(TILT_ANGLE)
    pos_labels = [f"pos {p} ({angles[p,0]:+.3f}, {angles[p,1]:+.3f})" for p in range(4)]

    # ── Figure 1: centroid scatter (2×2) ──
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    for p in range(4):
        ax = axes1[p // 2][p % 2]
        cx = np.array(data[p]["cx"])
        cy = np.array(data[p]["cy"])
        ax.scatter(cx - cx.mean(), cy - cy.mean(), s=2, alpha=0.4)
        ax.axhline(0, color="k", alpha=0.2)
        ax.axvline(0, color="k", alpha=0.2)
        ax.plot(0, 0, "r+", markersize=12, markeredgewidth=2)
        std_x, std_y = cx.std(), cy.std()
        ax.set_title(f"{pos_labels[p]}\nσx={std_x:.3f} σy={std_y:.3f} px")
        ax.set_xlabel("Δx (px)")
        ax.set_ylabel("Δy (px)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig1.suptitle("Centroid scatter (relative to mean)", fontsize=14)
    plt.tight_layout()
    fig1.savefig(str(out / "scatter.png"), dpi=150)

    # ── Figure 2: time series (2×2, each with x and y subplots) ──
    fig2, axes2 = plt.subplots(4, 2, figsize=(14, 12), sharex="col")
    for p in range(4):
        t = np.array(data[p]["time_ms"])
        t = t - t[0]  # relative time
        cx = np.array(data[p]["cx"])
        cy = np.array(data[p]["cy"])

        ax_x = axes2[p][0]
        ax_y = axes2[p][1]
        ax_x.plot(t, cx - cx.mean(), linewidth=0.5)
        ax_y.plot(t, cy - cy.mean(), linewidth=0.5)
        ax_x.set_ylabel(f"pos{p} Δx (px)")
        ax_y.set_ylabel(f"pos{p} Δy (px)")
        ax_x.grid(True, alpha=0.3)
        ax_y.grid(True, alpha=0.3)
    axes2[-1][0].set_xlabel("Time (ms)")
    axes2[-1][1].set_xlabel("Time (ms)")
    fig2.suptitle("Centroid drift over time", fontsize=14)
    plt.tight_layout()
    fig2.savefig(str(out / "timeseries.png"), dpi=150)

    # ── Figure 3: summary bar chart ──
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(4)
    bar_w = 0.25
    std_xs = [np.array(data[p]["cx"]).std() for p in range(4)]
    std_ys = [np.array(data[p]["cy"]).std() for p in range(4)]
    jitter = [np.sqrt(sx**2 + sy**2) for sx, sy in zip(std_xs, std_ys)]

    ax3.bar(x_pos - bar_w, std_xs, bar_w, label="σx")
    ax3.bar(x_pos, std_ys, bar_w, label="σy")
    ax3.bar(x_pos + bar_w, jitter, bar_w, label="total jitter")

    for i, j in enumerate(jitter):
        fps = data[i]["fps"]
        ax3.annotate(f"{j:.3f} px\n{fps:.0f} FPS",
                     (x_pos[i] + bar_w, j), textcoords="offset points",
                     xytext=(5, 5), fontsize=9)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"pos {p}" for p in range(4)])
    ax3.set_ylabel("Std deviation (px)")
    ax3.set_title("Position stability summary")
    ax3.legend()
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
        cam.exposure = EXPOSURE
        cam.gain = GAIN

        angles = XPRController.get_xpr_angles(TILT_ANGLE)

        params = {
            "tilt_angle": TILT_ANGLE,
            "num_frames": NUM_FRAMES,
            "exposure_us": cam.exposure,
            "gain_db": cam.gain,
            "roi_size": ROI_SIZE,
            "camera": {"width": cam.width, "height": cam.height, "is_color": cam.is_color},
            "timestamp": run_ts,
        }
        (out / "params.json").write_text(json.dumps(params, indent=2))

        print(f"Saving to {out}")
        print(f"Exposure: {cam.exposure} µs, Gain: {cam.gain} dB")
        print(f"Camera: {cam.width}x{cam.height}")
        print(f"Tilt angle: {TILT_ANGLE} rad, {NUM_FRAMES} frames per position")
        print()

        data = {}  # data[pos] = {"cx": [...], "cy": [...], "time_ms": [...], "fps": float}

        for p in range(4):
            pos_dir = out / f"pos{p}"
            pos_dir.mkdir()

            xpr.set_angles(angles[p, 0], angles[p, 1])
            time.sleep(0.02)  # 20ms settling

            # burst capture — store images in memory, process later
            print(f"  pos{p}: capturing {NUM_FRAMES} frames...", end=" ", flush=True)
            images = []
            timestamps = []
            t0 = time.perf_counter()
            for f in range(NUM_FRAMES):
                img = cam.capture_raw()
                timestamps.append((time.perf_counter() - t0) * 1000)  # ms
                images.append(img)
            elapsed = timestamps[-1]
            fps = NUM_FRAMES / (elapsed / 1000)
            print(f"{fps:.1f} FPS ({elapsed:.0f} ms)")

            # save images + find centroids
            print(f"         processing...", end=" ", flush=True)
            cxs, cys = [], []
            for f, img in enumerate(images):
                cv2.imwrite(str(pos_dir / f"frame_{f:03d}.png"), img)
                cx, cy = find_pinhole_center(img)
                cxs.append(cx)
                cys.append(cy)

            data[p] = {"cx": cxs, "cy": cys, "time_ms": timestamps, "fps": fps}
            std_x, std_y = np.std(cxs), np.std(cys)
            print(f"σx={std_x:.3f} σy={std_y:.3f} px")

        xpr.set_home()

    # save CSV
    csv_path = out / "centers.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "frame", "cx", "cy", "time_ms"])
        for p in range(4):
            for f in range(NUM_FRAMES):
                writer.writerow([p, f, data[p]["cx"][f], data[p]["cy"][f], data[p]["time_ms"][f]])
    print(f"\nCenters saved to {csv_path}")

    # save JSON summary
    summary = {}
    for p in range(4):
        cx, cy = np.array(data[p]["cx"]), np.array(data[p]["cy"])
        summary[f"pos{p}"] = {
            "cx_mean": float(cx.mean()), "cy_mean": float(cy.mean()),
            "cx_std": float(cx.std()), "cy_std": float(cy.std()),
            "jitter_px": float(np.sqrt(cx.std()**2 + cy.std()**2)),
            "fps": data[p]["fps"],
            "num_frames": NUM_FRAMES,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_results(data, out)
    print("Done.")


if __name__ == "__main__":
    run()
