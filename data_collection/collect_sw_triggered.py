"""
Collect raw images for super-resolution experiments.

Uses the same 9-position grid as the pinhole stability experiment:

    (-dx, +dy)  (0, +dy)  (+dx, +dy)
    (-dx,   0)  (0,   0)  (+dx,   0)
    (-dx, -dy)  (0, -dy)  (+dx, -dy)

X and Y axes are swept independently. For each tilt value, all 9 positions
are captured 5 times. A calibration CSV from the pinhole experiment is used
to tag each image with the expected pixel shift.

Edit the parameters below, then run:
    python collect_sw_triggered.py \
        --xpr-port /dev/ttyACM1 \
        --calibration ../calibration_beam_shift/data/shifts.csv
"""

import argparse
import csv
import json
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api import XPRController, DahengCamera

# ── Parameters ──────────────────────────────────────────────────────────────
TILT_X_MIN = 0.02              # degrees
TILT_X_MAX = 0.30
TILT_X_STEPS = 15

TILT_Y_MIN = 0.02              # degrees
TILT_Y_MAX = 0.30
TILT_Y_STEPS = 15

SETTLING_TIME_MS = 20          # ms, delay after each shift
NUM_REPEATS = 5                # captures per position
GAIN = 0                       # dB
EXPOSURE = None                # µs, or None for auto
OUTPUT_DIR = Path("./raw_capture_data")
# ────────────────────────────────────────────────────────────────────────────

TILT_X_ANGLES = np.linspace(TILT_X_MIN, TILT_X_MAX, TILT_X_STEPS)
TILT_Y_ANGLES = np.linspace(TILT_Y_MIN, TILT_Y_MAX, TILT_Y_STEPS)

GRID_SIGNS = [(sx, sy) for sy in [1, 0, -1] for sx in [-1, 0, 1]]
GRID_LABELS = [
    "(-x,+y)", "(0,+y)", "(+x,+y)",
    "(-x, 0)", "(0, 0)", "(+x, 0)",
    "(-x,-y)", "(0,-y)", "(+x,-y)",
]
CENTER_IDX = 4


def load_calibration(csv_path: str) -> dict:
    """Load pinhole calibration CSV into a lookup dict.

    Returns: {(sweep_axis, tilt_angle_str, position): (dx_mean, dy_mean), ...}
    """
    cal = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["sweep_axis"], row["tilt_angle_deg"], int(row["position"]))
            cal[key] = (float(row["dx_mean_px"]), float(row["dy_mean_px"]))
    return cal


def lookup_expected_shift(cal: dict, sweep_axis: str, tilt: float, pos: int) -> tuple[float, float]:
    """Find the closest calibration entry for a given tilt angle and position."""
    tilt_str = f"{tilt:.5f}"
    key = (sweep_axis, tilt_str, pos)
    if key in cal:
        return cal[key]

    # find closest tilt angle in calibration data
    cal_tilts = sorted(set(
        float(k[1]) for k in cal if k[0] == sweep_axis and k[2] == pos
    ))
    if not cal_tilts:
        return (0.0, 0.0)
    closest = min(cal_tilts, key=lambda t: abs(t - tilt))
    return cal.get((sweep_axis, f"{closest:.5f}", pos), (0.0, 0.0))


def _get_grid_positions(dx: float, dy: float) -> list[tuple[float, float]]:
    return [(sx * dx, sy * dy) for sx, sy in GRID_SIGNS]


def run_sweep(xpr, cam, tilt_angles, sweep_axis: str, cal: dict, out: Path):
    """Capture images for one axis sweep."""
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
        combo_label = f"sweep{sweep_axis}_tilt{tilt:.5f}deg"
        combo_dir = out / combo_label
        combo_dir.mkdir(exist_ok=True)
        print(f"\n  [{ti + 1}/{total_angles}] {sweep_axis}-sweep tilt = {tilt:.5f} deg")

        for r in range(NUM_REPEATS):
            for p, (ax, ay) in enumerate(positions):
                xpr.set_angles(ax, ay)
                time.sleep(SETTLING_TIME_MS / 1000.0)
                img = cam.capture_raw()
                capture_count += 1

                fname = f"{combo_label}/pos{p}_rep{r:02d}.png"
                cv2.imwrite(str(out / fname), img)

                # look up expected shift from calibration
                if p == CENTER_IDX:
                    exp_dx, exp_dy = 0.0, 0.0
                else:
                    exp_dx, exp_dy = lookup_expected_shift(cal, sweep_axis, tilt, p)

                saved_images.append({
                    "path": fname,
                    "sweep_axis": sweep_axis,
                    "tilt_deg": float(tilt),
                    "position": p,
                    "label": GRID_LABELS[p],
                    "commanded_angle_x_deg": float(ax),
                    "commanded_angle_y_deg": float(ay),
                    "repeat": r,
                    "expected_dx_px": exp_dx,
                    "expected_dy_px": exp_dy,
                })

                print(f"    rep {r+1}/{NUM_REPEATS} pos {p} {GRID_LABELS[p]:>8s}  "
                      f"cmd=({ax:+.5f},{ay:+.5f})  "
                      f"exp_shift=({exp_dx:+.3f},{exp_dy:+.3f})px  "
                      f"[{capture_count}/{total_captures}]")

        xpr.set_home()

    return saved_images


def run(xpr_port: str | None = None, calibration_csv: str | None = None):
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / run_ts
    out.mkdir(parents=True, exist_ok=True)

    # load calibration
    cal = {}
    if calibration_csv:
        print(f"Loading calibration from {calibration_csv}...")
        cal = load_calibration(calibration_csv)
        # show a few sample entries
        sample_keys = list(cal.keys())[:3]
        for k in sample_keys:
            print(f"  cal[{k}] = dx={cal[k][0]:+.3f}, dy={cal[k][1]:+.3f} px")
        print(f"Loaded {len(cal)} calibration entries.")
    else:
        print("WARNING: No calibration CSV provided. Expected shifts will be 0.")

    print("Connecting to XPR controller...")
    with XPRController(port=xpr_port) as xpr:
        print("XPR connected.")
        print("Connecting to camera...")
        with DahengCamera() as cam:
            print("Camera connected.")
            cam.gain = GAIN
            if EXPOSURE is not None:
                cam.exposure = EXPOSURE
            else:
                print("Running auto-exposure...")
                cam.auto_exposure()
                print(f"Auto-exposure done: {cam.exposure} µs")

            print(f"\nSaving to {out}")
            print(f"Exposure: {cam.exposure} µs, Gain: {cam.gain} dB")
            print(f"Camera: {cam.width}x{cam.height}, color={cam.is_color}")
            print(f"X tilt angles: {len(TILT_X_ANGLES)} steps [{TILT_X_MIN:.4f} .. {TILT_X_MAX:.4f}] deg")
            print(f"Y tilt angles: {len(TILT_Y_ANGLES)} steps [{TILT_Y_MIN:.4f} .. {TILT_Y_MAX:.4f}] deg")
            print(f"Settling time: {SETTLING_TIME_MS} ms, Repeats: {NUM_REPEATS}")
            print(f"Grid: 9 positions per tilt value")
            total = (len(TILT_X_ANGLES) + len(TILT_Y_ANGLES)) * 9 * NUM_REPEATS
            print(f"Total captures: {total}")
            print()

            # X-axis sweep
            print("=== X-axis sweep ===")
            x_images = run_sweep(xpr, cam, TILT_X_ANGLES, "x", cal, out)

            # Y-axis sweep
            print("\n=== Y-axis sweep ===")
            y_images = run_sweep(xpr, cam, TILT_Y_ANGLES, "y", cal, out)

            all_images = x_images + y_images

            # save results JSON
            results = {
                "description": (
                    "Raw image collection for super-resolution experiments. "
                    "A 3x3 grid of XPR positions is captured at each tilt angle, "
                    "with X and Y axes swept independently. Each position is captured "
                    f"{NUM_REPEATS} times. Expected pixel shifts are derived from "
                    "pinhole calibration data."
                ),
                "params": {
                    "tilt_x_angles_deg": TILT_X_ANGLES.tolist(),
                    "tilt_y_angles_deg": TILT_Y_ANGLES.tolist(),
                    "settling_time_ms": SETTLING_TIME_MS,
                    "num_repeats": NUM_REPEATS,
                    "exposure_us": cam.exposure,
                    "gain_db": cam.gain,
                    "camera": {"width": cam.width, "height": cam.height, "is_color": cam.is_color},
                    "calibration_csv": calibration_csv,
                },
                "grid_labels": GRID_LABELS,
                "timestamp": run_ts,
                "images": all_images,
            }
            (out / "results.json").write_text(json.dumps(results, indent=2))
            print(f"\nResults JSON saved to {out / 'results.json'}")

            # save image manifest CSV
            csv_path = out / "images.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["path", "sweep_axis", "tilt_deg", "position", "label",
                                 "commanded_angle_x_deg", "commanded_angle_y_deg",
                                 "repeat", "expected_dx_px", "expected_dy_px"])
                for img in all_images:
                    writer.writerow([
                        img["path"], img["sweep_axis"], img["tilt_deg"],
                        img["position"], img["label"],
                        img["commanded_angle_x_deg"], img["commanded_angle_y_deg"],
                        img["repeat"], img["expected_dx_px"], img["expected_dy_px"],
                    ])
            print(f"Image manifest CSV saved to {csv_path}")

    print(f"\nDone. {len(all_images)} images saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect raw images for SR experiments")
    parser.add_argument("--xpr-port", default=None,
                        help="Serial port for XPR controller (e.g. /dev/ttyACM1)")
    parser.add_argument("--calibration", default=None,
                        help="Path to pinhole experiment shifts.csv for expected pixel shifts")
    args = parser.parse_args()
    run(xpr_port=args.xpr_port, calibration_csv=args.calibration)
