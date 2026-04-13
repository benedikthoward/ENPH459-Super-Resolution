"""
Pinhole shift experiment: sweep tilt angles on each axis independently,
capture images of a pinhole at a 3x3 grid of positions, and measure
actual pixel shifts via 2D Gaussian PSF fitting.

For each tilt value, the 9-position grid is:

    (-dx, +dy)  (0, +dy)  (+dx, +dy)
    (-dx,   0)  (0,   0)  (+dx,   0)
    (-dx, -dy)  (0, -dy)  (+dx, -dy)

Position (0, 0) is the reference. X and Y axes are swept separately
(Y fixed at 0 during X sweep, X fixed at 0 during Y sweep).

Edit the parameters below, then run:
    uv run python -m optics_experiments.pinhole_stability --xpr-port /dev/ttyACM1
"""

import argparse
import csv
import json
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api import XPRController, DahengCamera
from data_collection.psf_mtf_utils import find_peak, extract_psf, fit_gaussian_psf

# ── Parameters ──────────────────────────────────────────────────────────────
TILT_X_MIN = 0.02              # degrees
TILT_X_MAX = 0.30
TILT_X_STEPS = 15

TILT_Y_MIN = 0.02              # degrees
TILT_Y_MAX = 0.30
TILT_Y_STEPS = 15

SETTLING_TIME_MS = 10          # ms, delay after each shift
NUM_REPEATS = 5                # repeats per tilt angle
GAIN = 0                       # dB
EXPOSURE = None                # µs, or None for auto
PSF_CROP_RADIUS = 50           # px, crop window around pinhole for PSF fit
OUTPUT_DIR = Path("./data")
# ────────────────────────────────────────────────────────────────────────────

TILT_X_ANGLES = np.linspace(TILT_X_MIN, TILT_X_MAX, TILT_X_STEPS)
TILT_Y_ANGLES = np.linspace(TILT_Y_MIN, TILT_Y_MAX, TILT_Y_STEPS)

# 9-position grid: (sx, sy) sign multipliers
# Row order: top (+y), middle (0), bottom (-y)
# Col order: left (-x), center (0), right (+x)
GRID_SIGNS = [(sx, sy) for sy in [1, 0, -1] for sx in [-1, 0, 1]]
GRID_LABELS = [
    "(-x,+y)", "(0,+y)", "(+x,+y)",
    "(-x, 0)", "(0, 0)", "(+x, 0)",
    "(-x,-y)", "(0,-y)", "(+x,-y)",
]
CENTER_IDX = 4  # index of (0, 0) in GRID_SIGNS


def find_pinhole_center(img: np.ndarray, crop_radius: int = PSF_CROP_RADIUS) -> tuple[float, float]:
    """Find sub-pixel (cx, cy) of a bright pinhole via 2D Gaussian PSF fit.
    Falls back to center-of-mass if the Gaussian fit fails."""
    gray = img.astype(np.float64) if img.dtype != np.float64 else img

    peak_r, peak_c = find_peak(gray)
    psf = extract_psf(gray, (peak_r, peak_c), crop_radius)

    # ROI origin in image coordinates
    h, w = gray.shape
    roi_r0 = max(peak_r - crop_radius, 0)
    roi_c0 = max(peak_c - crop_radius, 0)

    popt, _ = fit_gaussian_psf(psf)
    if popt is not None:
        # popt: [amp, x0, y0, sigma_x, sigma_y, theta, offset]
        cx = popt[1] + roi_c0
        cy = popt[2] + roi_r0
        return cx, cy

    # fallback: center-of-mass via cv2.moments
    bg = np.median(psf)
    threshed = np.clip(psf - bg, 0, None)
    thresh = threshed.max() * 0.1
    threshed[threshed < thresh] = 0
    M = cv2.moments(threshed)
    if M["m00"] == 0:
        return float(peak_c), float(peak_r)
    cx = M["m10"] / M["m00"] + roi_c0
    cy = M["m01"] / M["m00"] + roi_r0
    return cx, cy


def _get_grid_positions(dx: float, dy: float) -> list[tuple[float, float]]:
    """Return 9 (angle_x, angle_y) positions for given dx, dy magnitudes."""
    return [(sx * dx, sy * dy) for sx, sy in GRID_SIGNS]


def run_sweep(xpr, cam, tilt_angles, sweep_axis: str, out: Path):
    """Run a single-axis sweep. sweep_axis is 'x' or 'y'.

    For X sweep: dx varies, dy=0.
    For Y sweep: dy varies, dx=0.
    """
    results = {}
    csv_rows = []
    saved_images = []
    total_angles = len(tilt_angles)
    total_captures = total_angles * NUM_REPEATS * 9
    capture_count = 0

    for ti, tilt in enumerate(tilt_angles):
        if sweep_axis == "x":
            dx, dy = tilt, 0.0
        else:
            dx, dy = 0.0, tilt

        positions = _get_grid_positions(dx, dy)
        shifts_all_repeats = []
        combo_label = f"sweep{sweep_axis}_tilt{tilt:.5f}deg"
        combo_dir = out / combo_label
        combo_dir.mkdir(exist_ok=True)
        print(f"\n  [{ti + 1}/{total_angles}] {sweep_axis}-sweep tilt = {tilt:.5f} deg")

        for r in range(NUM_REPEATS):
            centers = {}
            for p, (ax, ay) in enumerate(positions):
                xpr.set_angles(ax, ay)
                time.sleep(SETTLING_TIME_MS / 1000.0)
                img = cam.capture_raw()
                capture_count += 1

                # save image on first repeat only
                if r == 0:
                    fname = f"{combo_label}/pos{p}_{GRID_LABELS[p].replace(' ','')}.png"
                    cv2.imwrite(str(out / fname), img)
                    saved_images.append(fname)

                cx, cy = find_pinhole_center(img)
                centers[p] = (cx, cy)
                csv_rows.append([sweep_axis, tilt, r, p, GRID_LABELS[p], ax, ay, cx, cy])

                print(f"    rep {r+1}/{NUM_REPEATS} pos {p} {GRID_LABELS[p]:>8s}  "
                      f"cx={cx:.2f} cy={cy:.2f}  "
                      f"[{capture_count}/{total_captures}]")

            # shifts relative to center position (0, 0)
            ref_cx, ref_cy = centers[CENTER_IDX]
            shifts = {}
            for p in range(9):
                if p == CENTER_IDX:
                    continue
                shifts[p] = {
                    "pos": p,
                    "label": GRID_LABELS[p],
                    "dx": centers[p][0] - ref_cx,
                    "dy": centers[p][1] - ref_cy,
                }
            shifts_all_repeats.append(shifts)

        # mean shifts across repeats
        mean_shifts = {}
        for p in range(9):
            if p == CENTER_IDX:
                continue
            dxs = [shifts_all_repeats[r][p]["dx"] for r in range(NUM_REPEATS)]
            dys = [shifts_all_repeats[r][p]["dy"] for r in range(NUM_REPEATS)]
            mean_shifts[p] = {
                "pos": p,
                "label": GRID_LABELS[p],
                "dx_mean": float(np.mean(dxs)),
                "dx_std": float(np.std(dxs)),
                "dy_mean": float(np.mean(dys)),
                "dy_std": float(np.std(dys)),
            }

        results[float(tilt)] = {
            "tilt_angle": float(tilt),
            "sweep_axis": sweep_axis,
            "mean_shifts": mean_shifts,
        }

        print(f"    Summary for tilt={tilt:.5f}:")
        for p, s in mean_shifts.items():
            print(f"      {s['label']}: dx={s['dx_mean']:+.3f}±{s['dx_std']:.3f}  "
                  f"dy={s['dy_mean']:+.3f}±{s['dy_std']:.3f} px")

        xpr.set_home()

    return results, csv_rows, saved_images


def plot_results(x_results: dict, y_results: dict, out: Path):
    # consistent colors/markers for each grid position
    pos_indices = [p for p in range(9) if p != CENTER_IDX]
    colors = plt.cm.tab10(np.linspace(0, 1, len(pos_indices)))
    pos_colors = {p: colors[i] for i, p in enumerate(pos_indices)}
    pos_markers = {p: m for p, m in zip(pos_indices, ["o", "s", "^", "D", "v", "<", ">", "p"])}

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 2, 2])

    # ── Position diagram (left column, spans both rows) ──
    ax_diagram = fig.add_subplot(gs[:, 0])
    ax_diagram.set_xlim(-1.5, 1.5)
    ax_diagram.set_ylim(-1.5, 1.5)
    ax_diagram.set_aspect("equal")
    ax_diagram.set_title("Position Grid Key")
    ax_diagram.set_xlabel("X sign")
    ax_diagram.set_ylabel("Y sign")

    for p, (sx, sy) in enumerate(GRID_SIGNS):
        if p == CENTER_IDX:
            ax_diagram.plot(sx, sy, marker="x", color="gray", markersize=15, markeredgewidth=3)
            ax_diagram.annotate("REF", (sx, sy), textcoords="offset points",
                                xytext=(12, -5), fontsize=9, color="gray")
        else:
            ax_diagram.plot(sx, sy, marker=pos_markers[p], color=pos_colors[p],
                            markersize=12, markeredgewidth=2)
            ax_diagram.annotate(GRID_LABELS[p], (sx, sy), textcoords="offset points",
                                xytext=(12, -5), fontsize=8, color=pos_colors[p])

    ax_diagram.grid(True, alpha=0.3)
    ax_diagram.set_xticks([-1, 0, 1])
    ax_diagram.set_yticks([-1, 0, 1])

    # ── X-sweep plots (top row) ──
    ax_dx_x = fig.add_subplot(gs[0, 1])
    ax_dy_x = fig.add_subplot(gs[0, 2])
    _plot_sweep(ax_dx_x, ax_dy_x, x_results, "X-sweep", pos_colors, pos_markers)

    # ── Y-sweep plots (bottom row) ──
    ax_dx_y = fig.add_subplot(gs[1, 1])
    ax_dy_y = fig.add_subplot(gs[1, 2])
    _plot_sweep(ax_dx_y, ax_dy_y, y_results, "Y-sweep", pos_colors, pos_markers)

    plt.tight_layout()
    fig.savefig(str(out / "pinhole_plots.png"), dpi=150)
    plt.show()
    print(f"Plots saved to {out / 'pinhole_plots.png'}")


def _plot_sweep(ax_dx, ax_dy, results: dict, title_prefix: str,
                pos_colors: dict, pos_markers: dict):
    tilt_angles = sorted(results.keys())

    for p in range(9):
        if p == CENTER_IDX:
            continue

        dx_means = [results[t]["mean_shifts"][p]["dx_mean"] for t in tilt_angles]
        dx_stds = [results[t]["mean_shifts"][p]["dx_std"] for t in tilt_angles]
        dy_means = [results[t]["mean_shifts"][p]["dy_mean"] for t in tilt_angles]
        dy_stds = [results[t]["mean_shifts"][p]["dy_std"] for t in tilt_angles]

        ax_dx.errorbar(tilt_angles, dx_means, yerr=dx_stds,
                       fmt="-", marker=pos_markers[p], color=pos_colors[p],
                       label=GRID_LABELS[p], markersize=4, capsize=3)
        ax_dy.errorbar(tilt_angles, dy_means, yerr=dy_stds,
                       fmt="-", marker=pos_markers[p], color=pos_colors[p],
                       label=GRID_LABELS[p], markersize=4, capsize=3)

    for ax, comp in [(ax_dx, "dx"), (ax_dy, "dy")]:
        ax.axhline(y=0, color="k", alpha=0.2)
        ax.set_xlabel("Tilt angle (degrees)")
        ax.set_ylabel(f"{comp} shift (px)")
        ax.set_title(f"{title_prefix}: {comp} vs tilt angle")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)


def save_shifts_csv(x_results: dict, y_results: dict, out: Path):
    """Save mean shifts as a CSV for easy analysis."""
    csv_path = out / "shifts.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sweep_axis", "tilt_angle_deg", "position", "label",
                         "dx_mean_px", "dx_std_px", "dy_mean_px", "dy_std_px"])
        for axis_label, results in [("x", x_results), ("y", y_results)]:
            for tilt in sorted(results.keys()):
                for p, s in results[tilt]["mean_shifts"].items():
                    writer.writerow([
                        axis_label, f"{tilt:.5f}", p, s["label"],
                        f"{s['dx_mean']:.4f}", f"{s['dx_std']:.4f}",
                        f"{s['dy_mean']:.4f}", f"{s['dy_std']:.4f}",
                    ])
    print(f"Shifts CSV saved to {csv_path}")


def run(xpr_port: str | None = None, exposure: float | None = None, gain: float | None = None):
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / run_ts
    out.mkdir(parents=True, exist_ok=True)

    use_exposure = exposure if exposure is not None else EXPOSURE
    use_gain = gain if gain is not None else GAIN

    with XPRController(port=xpr_port) as xpr, DahengCamera() as cam:
        cam.gain = use_gain
        if use_exposure is not None:
            cam.exposure = use_exposure
        else:
            # auto-expose targeting peak intensity ~230/255
            TARGET_PEAK = 220
            cam.exposure = 10000  # start at 10ms
            print("Auto-exposing for peak intensity target...")
            for iteration in range(15):
                frame = cam.capture_raw()
                peak = int(frame.max())
                current_exp = cam.exposure
                if peak == 0:
                    cam.exposure = current_exp * 4
                    print(f"  iter {iteration}: peak={peak}, exp={current_exp:.0f} -> {cam.exposure:.0f} µs")
                    continue
                ratio = TARGET_PEAK / peak
                new_exp = current_exp * ratio
                new_exp = max(100, min(500000, new_exp))  # clamp to valid range
                cam.exposure = new_exp
                print(f"  iter {iteration}: peak={peak}/255, exp={current_exp:.0f} -> {new_exp:.0f} µs")
                if abs(peak - TARGET_PEAK) <= 10:
                    break
            # verify final
            frame = cam.capture_raw()
            print(f"  Final: peak={int(frame.max())}/255, exposure={cam.exposure:.0f} µs")

        print(f"Saving to {out}")
        print(f"Exposure: {cam.exposure} µs, Gain: {cam.gain} dB")
        print(f"Camera: {cam.width}x{cam.height}, color={cam.is_color}")
        print(f"X tilt angles: {len(TILT_X_ANGLES)} steps [{TILT_X_MIN:.4f} .. {TILT_X_MAX:.4f}] deg")
        print(f"Y tilt angles: {len(TILT_Y_ANGLES)} steps [{TILT_Y_MIN:.4f} .. {TILT_Y_MAX:.4f}] deg")
        print(f"Settling time: {SETTLING_TIME_MS} ms, Repeats: {NUM_REPEATS}")
        print(f"Grid: 9 positions per tilt value")
        total = (len(TILT_X_ANGLES) + len(TILT_Y_ANGLES)) * 9 * NUM_REPEATS
        print(f"Total captures: {total}")
        print()

        # X-axis sweep (dy=0)
        print("=== X-axis sweep ===")
        x_results, x_csv, x_images = run_sweep(xpr, cam, TILT_X_ANGLES, "x", out)

        # Y-axis sweep (dx=0)
        print("\n=== Y-axis sweep ===")
        y_results, y_csv, y_images = run_sweep(xpr, cam, TILT_Y_ANGLES, "y", out)

        # save raw centers CSV
        csv_path = out / "centers.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sweep_axis", "tilt_angle", "repeat", "position",
                             "label", "angle_x", "angle_y", "cx", "cy"])
            writer.writerows(x_csv + y_csv)
        print(f"\nCenters saved to {csv_path}")

        # save shifts CSV
        save_shifts_csv(x_results, y_results, out)

        # save JSON with results, params, description, and image list
        all_results = {
            "description": (
                "Pinhole shift experiment: measures pixel shift of a pinhole image "
                "as a function of XPR tilt angle. A 3x3 grid of positions is visited "
                "at each tilt value: (-dx,+dy), (0,+dy), (+dx,+dy), (-dx,0), (0,0), "
                "(+dx,0), (-dx,-dy), (0,-dy), (+dx,-dy). Position (0,0) is the "
                "reference. X and Y axes are swept independently. Sub-pixel center "
                "is found via 2D Gaussian PSF fit."
            ),
            "params": {
                "tilt_x_angles_deg": TILT_X_ANGLES.tolist(),
                "tilt_y_angles_deg": TILT_Y_ANGLES.tolist(),
                "settling_time_ms": SETTLING_TIME_MS,
                "num_repeats": NUM_REPEATS,
                "psf_crop_radius": PSF_CROP_RADIUS,
                "exposure_us": cam.exposure,
                "gain_db": cam.gain,
                "camera": {"width": cam.width, "height": cam.height, "is_color": cam.is_color},
            },
            "grid_labels": GRID_LABELS,
            "timestamp": run_ts,
            "images": x_images + y_images,
            "x_sweep": {f"{k:.5f}": v for k, v in x_results.items()},
            "y_sweep": {f"{k:.5f}": v for k, v in y_results.items()},
        }
        (out / "results.json").write_text(json.dumps(all_results, indent=2))
        print(f"Results JSON saved to {out / 'results.json'}")

    plot_results(x_results, y_results, out)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pinhole shift stability experiment")
    parser.add_argument("--xpr-port", default=None,
                        help="Serial port for XPR controller (e.g. /dev/ttyACM1)")
    parser.add_argument("--exposure", type=float, default=None,
                        help="Exposure time in µs (default: auto)")
    parser.add_argument("--gain", type=float, default=None,
                        help="Gain in dB (default: 0)")
    args = parser.parse_args()
    run(xpr_port=args.xpr_port, exposure=args.exposure, gain=args.gain)
