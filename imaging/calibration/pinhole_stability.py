"""
Pinhole reticle stability: sweep tilt angles, capture images of a pinhole,
track its centroid in (x, y) to measure actual pixel shifts.

Mono Daheng camera. Auto-exposure for low light.

Edit the parameters below, then run:
    uv run python -m imaging.calibration.pinhole_stability
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
TILT_ANGLE_MIN = 0.02
TILT_ANGLE_MAX = 0.30
TILT_ANGLE_STEPS = 15

SETTLING_TIME_MS = 5           # fixed settling time after each shift
NUM_REPEATS = 10               # repeats per tilt angle for statistics
GAIN = 0                       # dB
EXPOSURE = None                # µs, or None for auto
ROI_SIZE = 100                 # px, crop window around pinhole for centroid
OUTPUT_DIR = Path("./stability_data")
# ────────────────────────────────────────────────────────────────────────────

TILT_ANGLES = np.linspace(TILT_ANGLE_MIN, TILT_ANGLE_MAX, TILT_ANGLE_STEPS)


def find_pinhole_center(img: np.ndarray, roi_size: int = ROI_SIZE) -> tuple[float, float]:
    """Find sub-pixel (cx, cy) of a bright pinhole on a dark background.
    Crops around the brightest region to exclude far-field noise."""
    # blur to find rough peak location
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    mx, my = max_loc

    # crop ROI around peak
    h, w = img.shape
    half = roi_size // 2
    x1 = max(0, mx - half)
    y1 = max(0, my - half)
    x2 = min(w, mx + half)
    y2 = min(h, my + half)
    roi = img[y1:y2, x1:x2].astype(np.float64)

    # subtract background and threshold
    bg = np.median(roi)
    roi = np.clip(roi - bg, 0, None)
    thresh = roi.max() * 0.1
    roi[roi < thresh] = 0

    # centroid via moments
    M = cv2.moments(roi)
    if M["m00"] == 0:
        return float(mx), float(my)
    cx = M["m10"] / M["m00"] + x1
    cy = M["m01"] / M["m00"] + y1
    return cx, cy


def plot_results(results: dict, out: Path):
    tilt_angles = sorted(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for p_idx in range(3):
        pos = p_idx + 1
        dx_means = [results[t]["mean_shifts"][p_idx]["dx_mean"] for t in tilt_angles]
        dx_stds = [results[t]["mean_shifts"][p_idx]["dx_std"] for t in tilt_angles]
        dy_means = [results[t]["mean_shifts"][p_idx]["dy_mean"] for t in tilt_angles]
        dy_stds = [results[t]["mean_shifts"][p_idx]["dy_std"] for t in tilt_angles]

        axes[0].errorbar(tilt_angles, dx_means, yerr=dx_stds,
                         fmt="o-", label=f"pos {pos}", markersize=4, capsize=3)
        axes[1].errorbar(tilt_angles, dy_means, yerr=dy_stds,
                         fmt="o-", label=f"pos {pos}", markersize=4, capsize=3)

    for ax, label in zip(axes, ["dx (px)", "dy (px)"]):
        ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3)
        ax.axhline(y=-1.0, color="k", linestyle="--", alpha=0.3)
        ax.axhline(y=0, color="k", alpha=0.2)
        ax.set_xlabel("Tilt angle (rad)")
        ax.set_ylabel(label)
        ax.set_title(f"Pixel shift ({label}) vs tilt angle")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(out / "pinhole_plots.png"), dpi=150)
    plt.show()
    print(f"Plots saved to {out / 'pinhole_plots.png'}")


def run():
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / run_ts
    out.mkdir(parents=True, exist_ok=True)

    with XPRController() as xpr, DahengCamera() as cam:
        cam.gain = GAIN
        if EXPOSURE is not None:
            cam.exposure = EXPOSURE
        else:
            cam.auto_exposure()

        params = {
            "tilt_angles": TILT_ANGLES.tolist(),
            "settling_time_ms": SETTLING_TIME_MS,
            "num_repeats": NUM_REPEATS,
            "roi_size": ROI_SIZE,
            "exposure_us": cam.exposure,
            "gain_db": cam.gain,
            "camera": {"width": cam.width, "height": cam.height, "is_color": cam.is_color},
            "timestamp": run_ts,
        }
        (out / "params.json").write_text(json.dumps(params, indent=2))

        print(f"Saving to {out}")
        print(f"Exposure: {cam.exposure} µs, Gain: {cam.gain} dB")
        print(f"Camera: {cam.width}x{cam.height}, color={cam.is_color}")
        print(f"Tilt angles: {len(TILT_ANGLES)} steps [{TILT_ANGLE_MIN:.4f} .. {TILT_ANGLE_MAX:.4f}] rad")
        print(f"Settling time: {SETTLING_TIME_MS} ms (fixed)")
        print(f"Repeats per angle: {NUM_REPEATS}")
        print()

        results = {}
        csv_rows = []

        for ti, tilt in enumerate(TILT_ANGLES):
            angles = XPRController.get_xpr_angles(tilt)
            shifts_all_repeats = []

            for r in range(NUM_REPEATS):
                centers = {}
                for p in range(4):
                    xpr.set_angles(angles[p, 0], angles[p, 1])
                    time.sleep(SETTLING_TIME_MS / 1000.0)
                    img = cam.capture_raw()

                    fname = f"tilt{tilt:.5f}_rep{r:02d}_pos{p}.png"
                    cv2.imwrite(str(out / fname), img)

                    cx, cy = find_pinhole_center(img)
                    centers[p] = (cx, cy)
                    csv_rows.append([tilt, r, p, cx, cy])

                # shifts relative to position 0
                ref_cx, ref_cy = centers[0]
                shifts = []
                for p in range(1, 4):
                    dx = centers[p][0] - ref_cx
                    dy = centers[p][1] - ref_cy
                    shifts.append({"pos": p, "dx": dx, "dy": dy})
                shifts_all_repeats.append(shifts)

            # mean shifts across repeats
            mean_shifts = []
            for p_idx in range(3):
                dxs = [shifts_all_repeats[r][p_idx]["dx"] for r in range(NUM_REPEATS)]
                dys = [shifts_all_repeats[r][p_idx]["dy"] for r in range(NUM_REPEATS)]
                mean_shifts.append({
                    "pos": p_idx + 1,
                    "dx_mean": float(np.mean(dxs)),
                    "dx_std": float(np.std(dxs)),
                    "dy_mean": float(np.mean(dys)),
                    "dy_std": float(np.std(dys)),
                })

            results[float(tilt)] = {
                "tilt_angle": float(tilt),
                "mean_shifts": mean_shifts,
                "all_repeats": shifts_all_repeats,
            }

            print(f"  [{ti + 1}/{len(TILT_ANGLES)}] tilt={tilt:.5f}:")
            for s in mean_shifts:
                print(f"    pos{s['pos']}: dx={s['dx_mean']:+.3f}±{s['dx_std']:.3f}  "
                      f"dy={s['dy_mean']:+.3f}±{s['dy_std']:.3f} px")

            xpr.set_home()

        # save CSV
        csv_path = out / "centers.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tilt_angle", "repeat", "position", "cx", "cy"])
            writer.writerows(csv_rows)
        print(f"\nCenters saved to {csv_path}")

        # save JSON
        # convert float keys to strings for JSON
        json_results = {f"{k:.5f}": v for k, v in results.items()}
        (out / "shifts.json").write_text(json.dumps(json_results, indent=2))
        print(f"Shifts saved to {out / 'shifts.json'}")

    plot_results(results, out)
    print("Done.")


if __name__ == "__main__":
    run()
