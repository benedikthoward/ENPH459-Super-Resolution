"""
Collect raw images for super-resolution using hardware triggering.

Uses the 4-corner diagonal pattern optimal for Bayer SR:
    pos 0: (-x, +y)    pos 1: (+x, +y)
    pos 2: (-x, -y)    pos 3: (+x, -y)

Three capture modes (each independently skippable):

  --nominal     Use hardcoded XPR nominal values (0.14391 deg = 0.5px mono,
                0.28782 deg = 1.0px color). No calibration needed.

  --special     Per-corner interpolated tilts from calibration data.
                Pass one or more calibration dirs via --calibration.

  --tilt-range  Sweep a range of tilt angles (currently skipped by default).

Each folder gets a metadata.json with tilt angles, expected pixel shifts,
and commanded positions.

    uv run python -m optics_experiments.collect_raw_images_hw \\
        --xpr-port /dev/ttyACM1 \\
        --nominal --special \\
        --calibration stability_data/20260326_123627 \\
        --calibration stability_data/20260326_151341 \\
        --calibration stability_data/20260326_152815 \\
        --calibration stability_data/20260327_121155
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
# Nominal XPR tilt values (from Daheng Mono datasheet)
NOMINAL_TILT_MONO = 0.14391    # degrees — gives ~0.5 px shift
NOMINAL_TILT_COLOR = 0.28782   # degrees — gives ~1.0 px shift (2× mono)

# Tilt sweep ranges (for --tilt-range mode)
TILT_MIN_COLOR = 0.26          # degrees
TILT_MAX_COLOR = 0.36
TILT_MIN_MONO = 0.13
TILT_MAX_MONO = 0.18
TILT_STEPS = 6

SETTLING_TIMES_MS = [50]              # ms
TRIGGER_PULSE_US = 100         # µs, GPIO pulse width

NUM_REPEATS = 4                # captures per position per tilt
GAIN = 0                       # dB
EXPOSURE = None                # µs, or None for auto
OUTPUT_DIR = Path("./raw_capture_data")

TARGET_SHIFT_PX_COLOR = 1.0
TARGET_SHIFT_PX_MONO = 0.5     # half pixel for mono
# ────────────────────────────────────────────────────────────────────────────

# 4-corner pattern: (sign_x, sign_y)
CORNER_SIGNS = [(-1, +1), (+1, +1), (-1, -1), (+1, -1)]
CORNER_LABELS = ["(-x,+y)", "(+x,+y)", "(-x,-y)", "(+x,-y)"]

CORNER_TO_CAL_POS = {
    0: 0,   # (-x,+y) → grid pos 0
    1: 2,   # (+x,+y) → grid pos 2
    2: 6,   # (-x,-y) → grid pos 6
    3: 8,   # (+x,-y) → grid pos 8
}


def load_calibration(csv_path: str) -> dict:
    cal = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row["sweep_axis"], row["tilt_angle_deg"], int(row["position"]))
            cal[key] = (float(row["dx_mean_px"]), float(row["dy_mean_px"]))
    return cal


def interpolate_tilt_for_corner(csv_path: str, target_px: float,
                                corner_idx: int) -> tuple[float, float]:
    """Interpolate tilt_x and tilt_y for one corner to achieve target_px shift."""
    cal_pos = CORNER_TO_CAL_POS[corner_idx]
    tilts_x, shifts_x = [], []
    tilts_y, shifts_y = [], []

    with open(csv_path) as f:
        for row in csv.DictReader(f):
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

    exp_dx = 0.0
    cal_tilts_x = sorted(set(
        float(k[1]) for k in cal if k[0] == "x" and k[2] == cal_pos
    ))
    if cal_tilts_x:
        closest_x = min(cal_tilts_x, key=lambda t: abs(t - tilt_x))
        entry = cal.get(("x", f"{closest_x:.5f}", cal_pos))
        if entry:
            exp_dx = entry[0]

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


def _write_folder_metadata(folder: Path, meta: dict):
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2))


def run_capture(xpr, cam, positions: list[tuple[float, float]],
                settle_ms: float, out: Path, label: str,
                cam_type: str, extra_meta: dict | None = None):
    """Capture 4-corner pattern. Returns list of image dicts."""
    folder = out / label
    folder.mkdir(exist_ok=True)

    meta = {"camera_type": cam_type, "settling_time_ms": settle_ms,
            "positions": [
                {"index": c, "label": CORNER_LABELS[c],
                 "commanded_x_deg": float(ax), "commanded_y_deg": float(ay)}
                for c, (ax, ay) in enumerate(positions)
            ]}
    if extra_meta:
        meta.update(extra_meta)
    _write_folder_metadata(folder, meta)

    saved = []
    for r in range(NUM_REPEATS):
        for c, (ax, ay) in enumerate(positions):
            xpr.set_angles(ax, ay)
            time.sleep(settle_ms / 1000.0)
            xpr.send_trigger_pulse(TRIGGER_PULSE_US)
            img = cam.capture_raw()

            fname = f"{label}/corner{c}_rep{r:02d}.png"
            cv2.imwrite(str(out / fname), img)

            saved.append({
                "path": fname,
                "settling_time_ms": settle_ms,
                "corner": c,
                "label": CORNER_LABELS[c],
                "commanded_x_deg": float(ax),
                "commanded_y_deg": float(ay),
                "repeat": r,
            })

            print(f"    rep {r+1}/{NUM_REPEATS} {CORNER_LABELS[c]:>8s}  "
                  f"cmd=({ax:+.5f},{ay:+.5f})")

    xpr.set_home()
    return saved


def run(xpr_port: str | None = None,
        calibration_dirs: list[str] | None = None,
        do_nominal: bool = True,
        do_special: bool = True,
        do_tilt_range: bool = False):

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / run_ts
    out.mkdir(parents=True, exist_ok=True)

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

            # detect camera type
            if cam.is_color:
                cam_type = "color"
                target_shift = TARGET_SHIFT_PX_COLOR
                nominal_tilt = NOMINAL_TILT_COLOR
                tilt_min, tilt_max = TILT_MIN_COLOR, TILT_MAX_COLOR
            else:
                cam_type = "mono"
                target_shift = TARGET_SHIFT_PX_MONO
                nominal_tilt = NOMINAL_TILT_MONO
                tilt_min, tilt_max = TILT_MIN_MONO, TILT_MAX_MONO

            print(f"\nDetected {cam_type.upper()} camera "
                  f"({cam.width}x{cam.height}) → {target_shift} px shift target")
            print(f"Nominal tilt: {nominal_tilt:.5f} deg")
            print(f"Exposure: {cam.exposure} µs, Gain: {cam.gain} dB")
            print(f"Modes: nominal={do_nominal}, special={do_special}, "
                  f"tilt_range={do_tilt_range}")
            print(f"Settling times: {SETTLING_TIMES_MS} ms")
            print(f"Repeats: {NUM_REPEATS}")
            print(f"Saving to {out}\n")

            all_images = []

            # ── NOMINAL: hardcoded XPR values ───────────────────────────
            if do_nominal:
                print(f"{'='*60}")
                print(f"=== NOMINAL: {nominal_tilt:.5f} deg "
                      f"({target_shift:.1f} px target) ===")
                print(f"{'='*60}")

                positions = [(sx * nominal_tilt, sy * nominal_tilt)
                             for sx, sy in CORNER_SIGNS]

                for settle_ms in SETTLING_TIMES_MS:
                    label = f"{cam_type}_nominal_{target_shift:.1f}px_settle{settle_ms}ms"
                    print(f"\n  settle={settle_ms}ms")
                    imgs = run_capture(
                        xpr, cam, positions, settle_ms, out, label, cam_type,
                        extra_meta={
                            "mode": "nominal",
                            "nominal_tilt_deg": nominal_tilt,
                            "target_shift_px": target_shift,
                        })
                    for img in imgs:
                        img["mode"] = "nominal"
                        img["tilt_x_deg"] = nominal_tilt
                        img["tilt_y_deg"] = nominal_tilt
                    all_images += imgs

            # ── SPECIAL: per-corner interpolated from calibration ───────
            if do_special and calibration_dirs:
                for cal_dir_str in calibration_dirs:
                    cal_dir = Path(cal_dir_str)
                    cal_csv = cal_dir / "shifts.csv"
                    if not cal_csv.exists():
                        print(f"\nWARNING: {cal_csv} not found, skipping.")
                        continue

                    cal_name = cal_dir.name
                    cal = load_calibration(str(cal_csv))
                    print(f"\n{'='*60}")
                    print(f"=== SPECIAL ({cal_name}): "
                          f"{target_shift:.1f} px per-corner ===")
                    print(f"{'='*60}")

                    # interpolate per-corner tilts
                    special_corners = {}
                    for c in range(4):
                        tx, ty = interpolate_tilt_for_corner(
                            str(cal_csv), target_shift, c)
                        special_corners[c] = (tx, ty)
                        print(f"  {CORNER_LABELS[c]}: "
                              f"tilt_x={tx:.5f}, tilt_y={ty:.5f} deg")

                    positions = []
                    for c in range(4):
                        tx, ty = special_corners[c]
                        sx, sy = CORNER_SIGNS[c]
                        positions.append((sx * tx, sy * ty))

                    for settle_ms in SETTLING_TIMES_MS:
                        label = (f"{cam_type}_special_{cal_name}"
                                 f"_{target_shift:.1f}px_settle{settle_ms}ms")
                        print(f"\n  settle={settle_ms}ms")

                        corner_meta = {}
                        for c in range(4):
                            tx, ty = special_corners[c]
                            exp_dx, exp_dy = lookup_expected_shift(cal, tx, ty, c)
                            corner_meta[CORNER_LABELS[c]] = {
                                "tilt_x_deg": tx, "tilt_y_deg": ty,
                                "commanded_x_deg": float(positions[c][0]),
                                "commanded_y_deg": float(positions[c][1]),
                                "expected_dx_px": exp_dx,
                                "expected_dy_px": exp_dy,
                            }

                        imgs = run_capture(
                            xpr, cam, positions, settle_ms, out, label, cam_type,
                            extra_meta={
                                "mode": "special",
                                "calibration_dir": cal_dir_str,
                                "calibration_csv": str(cal_csv),
                                "target_shift_px": target_shift,
                                "corners": corner_meta,
                            })
                        for img in imgs:
                            img["mode"] = f"special_{cal_name}"
                            c = img["corner"]
                            tx, ty = special_corners[c]
                            img["tilt_x_deg"] = tx
                            img["tilt_y_deg"] = ty
                        all_images += imgs

            elif do_special and not calibration_dirs:
                print("\nWARNING: --special requested but no --calibration dirs provided.")

            # ── TILT RANGE: sweep of tilt angles ────────────────────────
            if do_tilt_range:
                tilt_angles = np.linspace(tilt_min, tilt_max, TILT_STEPS)
                print(f"\n{'='*60}")
                print(f"=== TILT RANGE: [{tilt_min:.4f} .. {tilt_max:.4f}] deg "
                      f"({TILT_STEPS} steps) ===")
                print(f"{'='*60}")

                for settle_ms in SETTLING_TIMES_MS:
                    for ti, tilt in enumerate(tilt_angles):
                        label = f"{cam_type}_tilt{tilt:.5f}deg_settle{settle_ms}ms"
                        print(f"\n  [{ti+1}/{len(tilt_angles)}] "
                              f"tilt={tilt:.5f} deg, settle={settle_ms}ms")
                        imgs = run_capture(
                            xpr, cam,
                            [(sx * tilt, sy * tilt) for sx, sy in CORNER_SIGNS],
                            settle_ms, out, label, cam_type,
                            extra_meta={
                                "mode": "tilt_range",
                                "tilt_deg": float(tilt),
                            })
                        for img in imgs:
                            img["mode"] = "tilt_range"
                            img["tilt_x_deg"] = float(tilt)
                            img["tilt_y_deg"] = float(tilt)
                        all_images += imgs

            # ── Save results ────────────────────────────────────────────
            modes_run = []
            if do_nominal:
                modes_run.append("nominal")
            if do_special and calibration_dirs:
                modes_run.append(f"special ({len(calibration_dirs)} calibrations)")
            if do_tilt_range:
                modes_run.append("tilt_range")

            results = {
                "description": (
                    f"Raw image collection (hardware triggered, 4-corner diagonal, "
                    f"{cam_type} camera). Modes: {', '.join(modes_run)}."
                ),
                "params": {
                    "settling_times_ms": SETTLING_TIMES_MS,
                    "trigger_pulse_us": TRIGGER_PULSE_US,
                    "num_repeats": NUM_REPEATS,
                    "exposure_us": cam.exposure,
                    "gain_db": cam.gain,
                    "camera": {
                        "width": cam.width, "height": cam.height,
                        "is_color": cam.is_color,
                    },
                    "hardware_trigger": True,
                    "camera_type": cam_type,
                    "target_shift_px": target_shift,
                    "nominal_tilt_deg": nominal_tilt,
                    "corner_pattern": CORNER_LABELS,
                },
                "modes": {
                    "nominal": do_nominal,
                    "special": do_special,
                    "tilt_range": do_tilt_range,
                    "calibration_dirs": calibration_dirs or [],
                },
                "timestamp": run_ts,
                "total_images": len(all_images),
                "images": all_images,
            }
            (out / "results.json").write_text(json.dumps(results, indent=2))
            print(f"\nResults JSON: {out / 'results.json'}")

            csv_path = out / "images.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["path", "mode", "camera_type",
                                 "tilt_x_deg", "tilt_y_deg",
                                 "settling_time_ms", "corner", "label",
                                 "commanded_x_deg", "commanded_y_deg", "repeat"])
                for img in all_images:
                    writer.writerow([
                        img["path"], img.get("mode", ""),
                        cam_type,
                        img.get("tilt_x_deg", ""), img.get("tilt_y_deg", ""),
                        img["settling_time_ms"],
                        img["corner"], img["label"],
                        img["commanded_x_deg"], img["commanded_y_deg"],
                        img["repeat"],
                    ])
            print(f"Images CSV: {csv_path}")

    print(f"\nDone. {len(all_images)} images saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect raw images (hardware triggered)")
    parser.add_argument("--xpr-port", default=None,
                        help="Serial port for XPR controller")
    parser.add_argument("--calibration", action="append", default=None,
                        help="Calibration directory (can be repeated). "
                             "Each must contain shifts.csv.")

    # Mode flags
    parser.add_argument("--nominal", action="store_true",
                        help="Capture with hardcoded nominal XPR tilt values")
    parser.add_argument("--special", action="store_true",
                        help="Capture with per-corner interpolated tilts from calibration")
    parser.add_argument("--tilt-range", action="store_true",
                        help="Capture across a sweep of tilt angles")
    parser.add_argument("--all-modes", action="store_true",
                        help="Enable all three modes")

    args = parser.parse_args()

    # If no mode flags given, default to nominal + special
    do_nominal = args.nominal or args.all_modes
    do_special = args.special or args.all_modes
    do_tilt_range = args.tilt_range or args.all_modes

    if not (do_nominal or do_special or do_tilt_range):
        print("No mode selected. Defaulting to --nominal --special")
        do_nominal = True
        do_special = True

    run(xpr_port=args.xpr_port,
        calibration_dirs=args.calibration,
        do_nominal=do_nominal,
        do_special=do_special,
        do_tilt_range=do_tilt_range)
