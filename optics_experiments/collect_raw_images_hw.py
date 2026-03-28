"""
Collect raw images for super-resolution using hardware triggering.

Uses the 4-corner diagonal pattern optimal for Bayer SR:
    pos 0: (-x, +y)    pos 1: (+x, +y)
    pos 2: (-x, -y)    pos 3: (+x, -y)

Runs at multiple settling times. Includes a "special" run where each corner's
tilt is independently interpolated from calibration data to give exactly
TARGET_SHIFT_PX pixel shift.

Each folder gets a metadata.json with tilt angles, expected pixel shifts,
and commanded positions.

    uv run python -m optics_experiments.collect_raw_images_hw \
        --xpr-port /dev/ttyACM1 \
        --calibration stability_data/20260327_121155/shifts.csv
"""

import argparse
import csv
import json
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from imaging import XPRController, DahengCamera, TRIGGER_LINE2

# ── Parameters ──────────────────────────────────────────────────────────────
TILT_MIN = 0.26                # degrees (same range for both axes)
TILT_MAX = 0.36
TILT_STEPS = 6

SETTLING_TIMES_MS = [5, 50, 500]    # ms, multiple settling times to compare
TRIGGER_PULSE_US = 100         # µs, GPIO pulse width

NUM_REPEATS = 5                # captures per position per tilt
GAIN = 0                       # dB
EXPOSURE = None                # µs, or None for auto
OUTPUT_DIR = Path("./raw_capture_data")

# ── Special run parameters ─────────────────────────────────────────────────
TARGET_SHIFT_PX_COLOR = 1.0    # full pixel for Bayer color cameras
TARGET_SHIFT_PX_MONO = 0.5     # half pixel for mono cameras
CALIBRATION_DIR = "stability_data/20260327_121155"  # folder with shifts.csv
# ────────────────────────────────────────────────────────────────────────────

TILT_ANGLES = np.linspace(TILT_MIN, TILT_MAX, TILT_STEPS)

# 4-corner pattern: (sign_x, sign_y)
CORNER_SIGNS = [(-1, +1), (+1, +1), (-1, -1), (+1, -1)]
CORNER_LABELS = ["(-x,+y)", "(+x,+y)", "(-x,-y)", "(+x,-y)"]

# mapping from corner index to calibration grid positions (in the 3x3 grid)
# used to look up expected shifts from the pinhole stability data
CORNER_TO_CAL_POS = {
    0: 0,   # (-x,+y) → grid pos 0
    1: 2,   # (+x,+y) → grid pos 2
    2: 6,   # (-x,-y) → grid pos 6
    3: 8,   # (+x,-y) → grid pos 8
}


def load_calibration(csv_path: str) -> dict:
    cal = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["sweep_axis"], row["tilt_angle_deg"], int(row["position"]))
            cal[key] = (float(row["dx_mean_px"]), float(row["dy_mean_px"]))
    return cal


def interpolate_tilt_for_corner(csv_path: str, target_px: float,
                                corner_idx: int) -> tuple[float, float]:
    """Interpolate tilt_x and tilt_y for one corner to achieve target_px shift.

    Uses x-sweep data for the dx component and y-sweep data for the dy component.
    Returns (tilt_x_deg, tilt_y_deg).
    """
    cal_pos = CORNER_TO_CAL_POS[corner_idx]

    # interpolate x-tilt from x-sweep (dx column)
    tilts_x, shifts_x = [], []
    # interpolate y-tilt from y-sweep (dy column)
    tilts_y, shifts_y = [], []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pos = int(row["position"])
            if pos != cal_pos:
                continue
            tilt = float(row["tilt_angle_deg"])
            if row["sweep_axis"] == "x":
                tilts_x.append(tilt)
                shifts_x.append(abs(float(row["dx_mean_px"])))
            elif row["sweep_axis"] == "y":
                tilts_y.append(tilt)
                shifts_y.append(abs(float(row["dy_mean_px"])))

    if not tilts_x or not tilts_y:
        raise ValueError(f"Missing calibration data for corner {corner_idx}")

    order_x = np.argsort(shifts_x)
    tilt_x = float(np.interp(target_px, np.array(shifts_x)[order_x],
                              np.array(tilts_x)[order_x]))

    order_y = np.argsort(shifts_y)
    tilt_y = float(np.interp(target_px, np.array(shifts_y)[order_y],
                              np.array(tilts_y)[order_y]))

    return tilt_x, tilt_y


def lookup_expected_shift(cal: dict, tilt_x: float, tilt_y: float,
                          corner_idx: int) -> tuple[float, float]:
    """Look up expected (dx, dy) for a corner from calibration data."""
    cal_pos = CORNER_TO_CAL_POS[corner_idx]

    # x component from x-sweep
    exp_dx = 0.0
    cal_tilts_x = sorted(set(
        float(k[1]) for k in cal if k[0] == "x" and k[2] == cal_pos
    ))
    if cal_tilts_x:
        closest_x = min(cal_tilts_x, key=lambda t: abs(t - tilt_x))
        entry = cal.get(("x", f"{closest_x:.5f}", cal_pos))
        if entry:
            exp_dx = entry[0]

    # y component from y-sweep
    exp_dy = 0.0
    cal_tilts_y = sorted(set(
        float(k[1]) for k in cal if k[0] == "y" and k[2] == cal_pos
    ))
    if cal_tilts_y:
        closest_y = min(cal_tilts_y, key=lambda t: abs(t - tilt_y))
        entry = cal.get(("y", f"{closest_y:.5f}", cal_pos))
        if entry:
            exp_dy = entry[1]

    return exp_dx, exp_dy


def _write_folder_metadata(folder: Path, tilt_x: float, tilt_y: float,
                           settle_ms: float, cal: dict,
                           positions: list[tuple[float, float]],
                           cam_type: str = "unknown"):
    meta = {
        "camera_type": cam_type,
        "tilt_x_deg": tilt_x,
        "tilt_y_deg": tilt_y,
        "settling_time_ms": settle_ms,
        "positions": [
            {"index": c, "label": CORNER_LABELS[c],
             "commanded_x_deg": float(ax), "commanded_y_deg": float(ay)}
            for c, (ax, ay) in enumerate(positions)
        ],
        "expected_shifts": {},
    }
    for c in range(4):
        exp_dx, exp_dy = lookup_expected_shift(cal, tilt_x, tilt_y, c)
        meta["expected_shifts"][CORNER_LABELS[c]] = {"dx_px": exp_dx, "dy_px": exp_dy}
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2))


def run_sweep(xpr, cam, tilt_x: float, tilt_y: float, settle_ms: float,
              cal: dict, out: Path, label: str, cam_type: str = "unknown"):
    """Capture 4-corner pattern at one tilt setting."""
    positions = [(sx * tilt_x, sy * tilt_y) for sx, sy in CORNER_SIGNS]

    folder = out / label
    folder.mkdir(exist_ok=True)
    _write_folder_metadata(folder, tilt_x, tilt_y, settle_ms, cal, positions, cam_type)

    saved_images = []
    for r in range(NUM_REPEATS):
        for c, (ax, ay) in enumerate(positions):
            xpr.set_angles(ax, ay)
            time.sleep(settle_ms / 1000.0)
            xpr.send_trigger_pulse(TRIGGER_PULSE_US)
            img = cam.capture_raw()

            fname = f"{label}/corner{c}_rep{r:02d}.png"
            cv2.imwrite(str(out / fname), img)

            exp_dx, exp_dy = lookup_expected_shift(cal, tilt_x, tilt_y, c)

            saved_images.append({
                "path": fname,
                "tilt_x_deg": tilt_x,
                "tilt_y_deg": tilt_y,
                "settling_time_ms": settle_ms,
                "corner": c,
                "label": CORNER_LABELS[c],
                "commanded_x_deg": float(ax),
                "commanded_y_deg": float(ay),
                "repeat": r,
                "expected_dx_px": exp_dx,
                "expected_dy_px": exp_dy,
            })

            print(f"    rep {r+1}/{NUM_REPEATS} {CORNER_LABELS[c]:>8s}  "
                  f"cmd=({ax:+.5f},{ay:+.5f})  "
                  f"exp=({exp_dx:+.3f},{exp_dy:+.3f})px")

    xpr.set_home()
    return saved_images


def run(xpr_port: str | None = None, calibration_csv: str | None = None):
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / run_ts
    out.mkdir(parents=True, exist_ok=True)

    if calibration_csv is None:
        calibration_csv = str(Path(CALIBRATION_DIR) / "shifts.csv")

    # load calibration (special corners computed after camera detection)
    cal = {}
    if Path(calibration_csv).exists():
        print(f"Loading calibration from {calibration_csv}...")
        cal = load_calibration(calibration_csv)
        print(f"Loaded {len(cal)} calibration entries.")
    else:
        print(f"WARNING: {calibration_csv} not found. No calibration / special run.")

    # auto-exposure before opening in hardware trigger mode
    auto_exp = EXPOSURE
    if EXPOSURE is None:
        print("Running auto-exposure...")
        cam_sw = DahengCamera(hardware_trigger=False)
        cam_sw.gain = GAIN
        auto_exp = cam_sw.auto_exposure()
        cam_sw.close()
        print(f"Auto-exposure: {auto_exp} µs")

    print("\nConnecting to hardware (hw trigger on Line2, GPIO0)...")
    with XPRController(port=xpr_port) as xpr:
        xpr.setup_trigger_output()
        with DahengCamera(hardware_trigger=True, trigger_line=TRIGGER_LINE2) as cam:
            cam.gain = GAIN
            cam.exposure = auto_exp

            # auto-detect camera type
            if cam.is_color:
                cam_type = "color"
                target_shift = TARGET_SHIFT_PX_COLOR
            else:
                cam_type = "mono"
                target_shift = TARGET_SHIFT_PX_MONO
            print(f"\nDetected {cam_type.upper()} camera "
                  f"({cam.width}x{cam.height}) → {target_shift} px shift target")

            # compute special corners now that we know the target shift
            special_corners = {}
            if cal:
                print(f"\nSpecial run: {target_shift:.1f} px target per corner")
                for c in range(4):
                    tx, ty = interpolate_tilt_for_corner(
                        calibration_csv, target_shift, c)
                    special_corners[c] = (tx, ty)
                    print(f"  {CORNER_LABELS[c]}: tilt_x={tx:.5f}, tilt_y={ty:.5f} deg")

            print(f"\nSaving to {out}")
            print(f"Exposure: {cam.exposure} µs, Gain: {cam.gain} dB")
            print(f"Trigger: hardware (Line2/GPIO0)")
            print(f"Settling times: {SETTLING_TIMES_MS} ms")
            print(f"Tilt range: {len(TILT_ANGLES)} steps [{TILT_MIN:.4f} .. {TILT_MAX:.4f}] deg")
            print(f"Repeats: {NUM_REPEATS}")
            total = len(SETTLING_TIMES_MS) * len(TILT_ANGLES) * NUM_REPEATS * 4
            if special_corners:
                total += len(SETTLING_TIMES_MS) * NUM_REPEATS * 4
            print(f"Total captures: ~{total}")

            all_images = []

            for settle_ms in SETTLING_TIMES_MS:
                print(f"\n{'='*60}")
                print(f"=== Settling time: {settle_ms} ms ===")
                print(f"{'='*60}")

                for ti, tilt in enumerate(TILT_ANGLES):
                    label = f"{cam_type}_tilt{tilt:.5f}deg_settle{settle_ms}ms"
                    print(f"\n  [{ti+1}/{len(TILT_ANGLES)}] tilt={tilt:.5f} deg")
                    all_images += run_sweep(xpr, cam, tilt, tilt, settle_ms,
                                            cal, out, label, cam_type)

            # special runs: per-corner interpolated tilts
            if special_corners:
                for settle_ms in SETTLING_TIMES_MS:
                    label = f"{cam_type}_special_{target_shift:.1f}px_settle{settle_ms}ms"
                    print(f"\n{'='*60}")
                    print(f"=== Special: {target_shift:.1f}px target ({cam_type}), "
                          f"settle={settle_ms}ms ===")
                    print(f"{'='*60}")

                    folder = out / label
                    folder.mkdir(exist_ok=True)

                    # build per-corner positions with independent tilts
                    positions = []
                    for c in range(4):
                        tx, ty = special_corners[c]
                        sx, sy = CORNER_SIGNS[c]
                        positions.append((sx * tx, sy * ty))

                    # write metadata with per-corner tilt info
                    meta = {
                        "camera_type": cam_type,
                        "target_shift_px": target_shift,
                        "settling_time_ms": settle_ms,
                        "corners": {},
                    }
                    for c in range(4):
                        tx, ty = special_corners[c]
                        exp_dx, exp_dy = lookup_expected_shift(cal, tx, ty, c)
                        meta["corners"][CORNER_LABELS[c]] = {
                            "tilt_x_deg": tx,
                            "tilt_y_deg": ty,
                            "commanded_x_deg": float(positions[c][0]),
                            "commanded_y_deg": float(positions[c][1]),
                            "expected_dx_px": exp_dx,
                            "expected_dy_px": exp_dy,
                        }
                    (folder / "metadata.json").write_text(json.dumps(meta, indent=2))

                    saved = []
                    for r in range(NUM_REPEATS):
                        for c, (ax, ay) in enumerate(positions):
                            xpr.set_angles(ax, ay)
                            time.sleep(settle_ms / 1000.0)
                            xpr.send_trigger_pulse(TRIGGER_PULSE_US)
                            img = cam.capture_raw()

                            fname = f"{label}/corner{c}_rep{r:02d}.png"
                            cv2.imwrite(str(out / fname), img)

                            tx, ty = special_corners[c]
                            exp_dx, exp_dy = lookup_expected_shift(cal, tx, ty, c)

                            entry = {
                                "path": fname,
                                "tilt_x_deg": tx,
                                "tilt_y_deg": ty,
                                "settling_time_ms": settle_ms,
                                "corner": c,
                                "label": CORNER_LABELS[c],
                                "commanded_x_deg": float(ax),
                                "commanded_y_deg": float(ay),
                                "repeat": r,
                                "expected_dx_px": exp_dx,
                                "expected_dy_px": exp_dy,
                                "special": True,
                            }
                            saved.append(entry)

                            print(f"    rep {r+1}/{NUM_REPEATS} {CORNER_LABELS[c]:>8s}  "
                                  f"tilt=({tx:.5f},{ty:.5f})  "
                                  f"cmd=({ax:+.5f},{ay:+.5f})  "
                                  f"exp=({exp_dx:+.3f},{exp_dy:+.3f})px")

                    xpr.set_home()
                    all_images += saved

            # save results
            results = {
                "description": (
                    f"Raw image collection (hardware triggered, 4-corner diagonal, {cam_type} camera). "
                    f"Settling times: {SETTLING_TIMES_MS} ms. "
                    f"Special runs at {target_shift:.1f} px target with per-corner "
                    f"independently interpolated tilts."
                ),
                "params": {
                    "tilt_angles_deg": TILT_ANGLES.tolist(),
                    "settling_times_ms": SETTLING_TIMES_MS,
                    "trigger_pulse_us": TRIGGER_PULSE_US,
                    "num_repeats": NUM_REPEATS,
                    "exposure_us": cam.exposure,
                    "gain_db": cam.gain,
                    "camera": {
                        "width": cam.width, "height": cam.height,
                        "is_color": cam.is_color,
                    },
                    "calibration_csv": calibration_csv,
                    "calibration_dir": CALIBRATION_DIR,
                    "hardware_trigger": True,
                    "camera_type": cam_type,
                    "target_shift_px": target_shift,
                    "special_corners": {
                        CORNER_LABELS[c]: {"tilt_x": tx, "tilt_y": ty}
                        for c, (tx, ty) in special_corners.items()
                    } if special_corners else None,
                    "corner_pattern": CORNER_LABELS,
                },
                "timestamp": run_ts,
                "images": all_images,
            }
            (out / "results.json").write_text(json.dumps(results, indent=2))

            csv_path = out / "images.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["path", "camera_type", "tilt_x_deg", "tilt_y_deg",
                                 "settling_time_ms", "corner", "label",
                                 "commanded_x_deg", "commanded_y_deg",
                                 "repeat", "expected_dx_px", "expected_dy_px"])
                for img in all_images:
                    writer.writerow([
                        img["path"], cam_type,
                        img["tilt_x_deg"], img["tilt_y_deg"],
                        img["settling_time_ms"],
                        img["corner"], img["label"],
                        img["commanded_x_deg"], img["commanded_y_deg"],
                        img["repeat"], img["expected_dx_px"], img["expected_dy_px"],
                    ])
            print(f"\nImages CSV: {csv_path}")

    print(f"\nDone. {len(all_images)} images saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect raw images (hardware triggered)")
    parser.add_argument("--xpr-port", default=None,
                        help="Serial port for XPR controller")
    parser.add_argument("--calibration", default=None,
                        help="Path to shifts.csv (default: CALIBRATION_DIR/shifts.csv)")
    args = parser.parse_args()
    run(xpr_port=args.xpr_port, calibration_csv=args.calibration)
