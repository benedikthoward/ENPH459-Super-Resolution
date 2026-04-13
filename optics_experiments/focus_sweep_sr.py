"""
Focus-sweep super-resolution data collection.

Captures a 9-position shifted image grid at multiple pixel-shift magnitudes
and focus positions, for evaluating SR reconstruction and barcode decoding
performance as a function of shift accuracy and defocus.

Grid pattern (same as pinhole stability):
    (-dx, +dy)  (0, +dy)  (+dx, +dy)
    (-dx,   0)  (0,   0)  (+dx,   0)
    (-dx, -dy)  (0, -dy)  (+dx, -dy)

Uses hardware triggering (XPR GPIO0 → Camera Line2).
Tilt angles are computed from pinhole calibration data, independently for
X and Y axes.

    uv run python -m optics_experiments.focus_sweep_sr \\
        --port /dev/ttyACM0 \\
        --xpr-port /dev/ttyACM1 \\
        --calibration stability_data/20260327_121155/shifts.csv \\
        --title barcode_test
"""

import argparse
import csv
import json
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from zaber_motion import Units
from zaber_motion.ascii import Connection

from imaging import ImagingSystem
from imaging.camera import DahengCamera

# ── Parameters ──────────────────────────────────────────────────────────────
TARGET_SHIFTS_MONO = [0.4, 0.45, 0.5, 0.55, 0.6]   # half-pixel for mono
TARGET_SHIFTS_COLOR = [0.8, 0.9, 1.0, 1.1, 1.2]   # full-pixel for color (Bayer)

FOCUS_RANGE_MM = 5.0           # ±mm from starting position
FOCUS_STEP_MM = 0.5            # mm between focus positions
FOCUS_SETTLE_S = 1.0           # seconds to wait after Zaber move

XPR_SETTLE_MS = 20             # ms to wait after XPR shift
NUM_REPEATS = 2                # repetitions per (shift, focus) combo
GAIN = 0                       # dB
EXPOSURE = None                # µs, or None for auto (target peak ~230)
TARGET_PEAK = 230              # target peak intensity for auto-exposure
TRIGGER_PULSE_US = 100         # µs, GPIO pulse width

OUTPUT_DIR = Path("./raw_capture_data")
# ────────────────────────────────────────────────────────────────────────────

GRID_SIGNS = [(sx, sy) for sy in [1, 0, -1] for sx in [-1, 0, 1]]
GRID_LABELS = [
    "(-x,+y)", "(0,+y)", "(+x,+y)",
    "(-x, 0)", "(0, 0)", "(+x, 0)",
    "(-x,-y)", "(0,-y)", "(+x,-y)",
]
CENTER_IDX = 4


def laplacian_variance(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def load_calibration(csv_path: str) -> dict:
    """Load pinhole calibration CSV.
    Returns: {(sweep_axis, tilt_angle_str, position): (dx_mean, dy_mean)}
    """
    cal = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row["sweep_axis"], row["tilt_angle_deg"], int(row["position"]))
            cal[key] = (float(row["dx_mean_px"]), float(row["dy_mean_px"]))
    return cal


def interpolate_tilt_for_shift(cal: dict, target_px: float) -> tuple[float, float]:
    """Compute (tilt_x_deg, tilt_y_deg) needed for a target pixel shift.

    Uses x-sweep position 5 (+x,0) for tilt_x and y-sweep position 7 (0,-y)
    for tilt_y — these are pure single-axis shifts.
    """
    # X axis: position 5 is (+x, 0) — pure dx
    tilts_x, shifts_x = [], []
    for (axis, tilt_str, pos), (dx, dy) in cal.items():
        if axis == "x" and pos == 5:
            tilts_x.append(float(tilt_str))
            shifts_x.append(abs(dx))

    # Y axis: position 7 is (0, -y) — pure dy
    tilts_y, shifts_y = [], []
    for (axis, tilt_str, pos), (dx, dy) in cal.items():
        if axis == "y" and pos == 7:
            tilts_y.append(float(tilt_str))
            shifts_y.append(abs(dy))

    if not tilts_x or not tilts_y:
        raise ValueError("Calibration data missing positions 5 (x-sweep) or 7 (y-sweep)")

    order_x = np.argsort(shifts_x)
    tilt_x = float(np.interp(target_px, np.array(shifts_x)[order_x],
                              np.array(tilts_x)[order_x]))

    order_y = np.argsort(shifts_y)
    tilt_y = float(np.interp(target_px, np.array(shifts_y)[order_y],
                              np.array(tilts_y)[order_y]))

    return tilt_x, tilt_y


def build_grid_positions(tilt_x: float, tilt_y: float) -> list[tuple[float, float]]:
    """Build the 9-position grid from tilt angles."""
    return [(sx * tilt_x, sy * tilt_y) for sx, sy in GRID_SIGNS]


def auto_expose_for_peak(target_peak: int = TARGET_PEAK, gain: float = GAIN,
                         max_iter: int = 10) -> tuple[float, bool]:
    """Binary-search exposure to get peak intensity near target_peak.
    Opens a software-triggered camera, finds exposure, closes it.
    Returns (exposure_us, is_color).
    """
    print("Auto-exposing for target peak intensity...")
    with DahengCamera(hardware_trigger=False) as cam:
        cam.gain = gain
        is_color = cam.is_color
        # start with SDK auto-exposure as initial guess
        cam.auto_exposure()
        exp = cam.exposure

        for i in range(max_iter):
            cam.exposure = exp
            img = cam.capture_raw()
            peak = int(img.max())
            print(f"  iter {i+1}: exposure={exp:.0f} µs, peak={peak}/255")

            if abs(peak - target_peak) <= 5:
                break
            if peak == 0:
                exp *= 4
                continue
            # scale exposure proportionally
            exp = exp * (target_peak / max(peak, 1))
            exp = max(100, min(500000, exp))

        final_exp = cam.exposure
        cam_type = "color" if is_color else "mono"
        print(f"  Final exposure: {final_exp:.0f} µs (peak={img.max()})")
        print(f"  Detected {cam_type} camera")
    return final_exp, is_color


def run(zaber_port: str, xpr_port: str | None = None,
        calibration_csv: str | None = None, title: str = "sr_sweep",
        only_optimal: bool = False):
    start_time = datetime.now()
    run_ts = start_time.strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"{run_ts}_{title}"
    out.mkdir(parents=True, exist_ok=True)

    # ── Load calibration ────────────────────────────────────────────────
    if calibration_csv is None:
        raise ValueError("--calibration is required (path to pinhole shifts.csv)")

    print(f"Loading calibration from {calibration_csv}...")
    cal = load_calibration(calibration_csv)
    print(f"  {len(cal)} entries loaded.")

    # ── Auto-expose and detect camera type ────────────────────────────
    if EXPOSURE is not None:
        exposure = EXPOSURE
        # still need to detect camera type
        with DahengCamera(hardware_trigger=False) as cam_tmp:
            is_color = cam_tmp.is_color
        print(f"\nUsing fixed exposure: {exposure} µs")
    else:
        exposure, is_color = auto_expose_for_peak()

    cam_type = "color" if is_color else "mono"
    all_shifts = TARGET_SHIFTS_COLOR if is_color else TARGET_SHIFTS_MONO
    if only_optimal:
        # just the middle (optimal) shift value
        target_shifts = [all_shifts[len(all_shifts) // 2]]
        print(f"  --only-optimal: using single shift {target_shifts[0]:.2f} px at in-focus position")
    else:
        target_shifts = all_shifts
    print(f"  Camera type: {cam_type}")
    print(f"  Target shifts: {target_shifts} px "
          f"({'full pixel for Bayer' if is_color else 'half pixel for mono'})")

    # Compute tilt angles for each target shift
    shift_configs = []
    print("\nTarget pixel shifts → tilt angles:")
    for target_px in target_shifts:
        tilt_x, tilt_y = interpolate_tilt_for_shift(cal, target_px)
        shift_configs.append({
            "target_px": target_px,
            "tilt_x_deg": tilt_x,
            "tilt_y_deg": tilt_y,
        })
        print(f"  {target_px:.2f} px → tilt_x={tilt_x:.5f}°, tilt_y={tilt_y:.5f}°")

    # ── Connect Zaber stage ─────────────────────────────────────────────
    print(f"\nConnecting to Zaber stage on {zaber_port}...")
    conn = Connection.open_serial_port(zaber_port)
    conn.enable_alerts()
    devices = conn.detect_devices()
    if not devices:
        raise RuntimeError("No Zaber devices found")
    device = devices[0]

    # Use X axis for focus
    try:
        x_axis = device.get_lockstep(1)
    except Exception:
        x_axis = device.get_axis(1)

    start_pos = x_axis.get_position(Units.LENGTH_MILLIMETRES)
    print(f"  Starting X position (in focus): {start_pos:.3f} mm")

    # Generate focus positions
    if only_optimal:
        focus_offsets = np.array([0.0])
        focus_positions = np.array([start_pos])
        print(f"  --only-optimal: single focus position at {start_pos:.3f} mm")
    else:
        focus_offsets = np.arange(-FOCUS_RANGE_MM, FOCUS_RANGE_MM + FOCUS_STEP_MM / 2,
                                  FOCUS_STEP_MM)
        focus_positions = start_pos + focus_offsets
    num_focus = len(focus_positions)
    if not only_optimal:
        print(f"  Focus sweep: {num_focus} positions, "
              f"[{focus_positions[0]:.1f} .. {focus_positions[-1]:.1f}] mm")

    # ── Connect imaging system (hardware trigger) ───────────────────────
    print(f"\nConnecting imaging system (hw trigger, XPR port={xpr_port})...")
    sys = ImagingSystem(hardware_trigger=True, xpr_port=xpr_port)
    sys.camera.gain = GAIN
    sys.camera.exposure = exposure
    cam_width = sys.camera.width
    cam_height = sys.camera.height
    cam_is_color = sys.camera.is_color
    print(f"  Camera: {cam_width}x{cam_height}, "
          f"exposure={sys.camera.exposure:.0f} µs, gain={sys.camera.gain:.1f} dB")

    total_images = num_focus * len(target_shifts) * NUM_REPEATS * 9
    total_with_ref = total_images + num_focus
    print(f"\n{'='*60}")
    print(f"  Experiment: {title}")
    print(f"  Focus positions: {num_focus}")
    print(f"  Shift magnitudes: {len(target_shifts)}")
    print(f"  Repeats: {NUM_REPEATS}")
    print(f"  Total captures: {total_with_ref}")
    print(f"  Output: {out}")
    print(f"{'='*60}\n")

    all_images = []
    focus_summary = []
    capture_count = 0

    try:
        for fi, (offset, focus_mm) in enumerate(zip(focus_offsets, focus_positions)):
            # ── Move to focus position ──────────────────────────────────
            print(f"\n[Focus {fi+1}/{num_focus}] offset={offset:+.1f}mm → {focus_mm:.3f}mm")
            x_axis.move_absolute(focus_mm, Units.LENGTH_MILLIMETRES, wait_until_idle=True)
            time.sleep(FOCUS_SETTLE_S)

            # Capture reference image at center for laplacian variance
            sys.xpr.set_home()
            time.sleep(XPR_SETTLE_MS / 1000.0)
            ref_img = sys._capture_frame()
            lap_var = laplacian_variance(ref_img)
            capture_count += 1
            print(f"  Laplacian variance: {lap_var:.1f}")

            # Folder name for this focus position
            dist_label = f"dist{offset:+.1f}mm_lap{lap_var:.1f}"
            dist_dir = out / dist_label
            dist_dir.mkdir(exist_ok=True)

            # Save reference image
            cv2.imwrite(str(dist_dir / "reference_center.png"), ref_img)

            focus_entry = {
                "focus_index": fi,
                "offset_mm": float(offset),
                "position_mm": float(focus_mm),
                "laplacian_variance": float(lap_var),
                "folder": dist_label,
            }
            focus_summary.append(focus_entry)

            # ── Sweep through shift magnitudes ──────────────────────────
            for si, cfg in enumerate(shift_configs):
                target_px = cfg["target_px"]
                tilt_x = cfg["tilt_x_deg"]
                tilt_y = cfg["tilt_y_deg"]
                positions = build_grid_positions(tilt_x, tilt_y)

                for rep in range(NUM_REPEATS):
                    cycle_label = (f"shift{tilt_x:.5f}deg_{target_px:.2f}px"
                                   f"_cycle{si}_rep{rep}")
                    cycle_dir = dist_dir / cycle_label
                    cycle_dir.mkdir(exist_ok=True)

                    # Per-folder metadata
                    folder_meta = {
                        "focus_offset_mm": float(offset),
                        "focus_position_mm": float(focus_mm),
                        "laplacian_variance": float(lap_var),
                        "target_shift_px": target_px,
                        "tilt_x_deg": tilt_x,
                        "tilt_y_deg": tilt_y,
                        "cycle_index": si,
                        "repetition": rep,
                        "positions": [
                            {"index": p, "label": GRID_LABELS[p],
                             "commanded_x_deg": float(ax),
                             "commanded_y_deg": float(ay)}
                            for p, (ax, ay) in enumerate(positions)
                        ],
                    }
                    (cycle_dir / "metadata.json").write_text(
                        json.dumps(folder_meta, indent=2))

                    for p, (ax, ay) in enumerate(positions):
                        sys.xpr.set_angles(ax, ay)
                        time.sleep(XPR_SETTLE_MS / 1000.0)
                        img = sys._capture_frame()
                        capture_count += 1

                        fname = f"pos{p}_{GRID_LABELS[p]}.png"
                        cv2.imwrite(str(cycle_dir / fname), img)

                        rel_path = f"{dist_label}/{cycle_label}/{fname}"
                        all_images.append({
                            "path": rel_path,
                            "focus_offset_mm": float(offset),
                            "focus_position_mm": float(focus_mm),
                            "laplacian_variance": float(lap_var),
                            "target_shift_px": target_px,
                            "tilt_x_deg": tilt_x,
                            "tilt_y_deg": tilt_y,
                            "cycle_index": si,
                            "repetition": rep,
                            "position": p,
                            "label": GRID_LABELS[p],
                            "commanded_x_deg": float(ax),
                            "commanded_y_deg": float(ay),
                        })

                    sys.xpr.set_home()

                    print(f"  shift {si+1}/{len(target_shifts)} "
                          f"({target_px:.2f}px) rep {rep+1}/{NUM_REPEATS}  "
                          f"[{capture_count}/{total_with_ref}]")

        # ── Return to start ─────────────────────────────────────────────
        print(f"\nReturning to starting position ({start_pos:.3f} mm)...")
        x_axis.move_absolute(start_pos, Units.LENGTH_MILLIMETRES, wait_until_idle=True)

    finally:
        sys.close()
        conn.close()

    end_time = datetime.now()
    duration = end_time - start_time

    # ── Save results.json ───────────────────────────────────────────────
    results = {
        "description": (
            "Focus-sweep super-resolution data collection. Captures a 3x3 grid "
            "of shifted images at multiple pixel-shift magnitudes and focus "
            "positions. Purpose: evaluate SR reconstruction quality and barcode "
            "decoding performance as a function of sub-pixel shift accuracy "
            "and defocus."
        ),
        "params": {
            "camera_type": cam_type,
            "target_shifts_px": target_shifts,
            "focus_range_mm": FOCUS_RANGE_MM,
            "focus_step_mm": FOCUS_STEP_MM,
            "focus_settle_s": FOCUS_SETTLE_S,
            "xpr_settle_ms": XPR_SETTLE_MS,
            "num_repeats": NUM_REPEATS,
            "gain_db": GAIN,
            "exposure_us": exposure,
            "target_peak_intensity": TARGET_PEAK,
            "trigger_pulse_us": TRIGGER_PULSE_US,
        },
        "calibration": {
            "csv_path": calibration_csv,
            "shift_configs": shift_configs,
        },
        "hardware": {
            "xpr_port": xpr_port,
            "zaber_port": zaber_port,
            "trigger": "XPR GPIO0 → Camera Line2 (rising edge)",
            "focus_axis": "X (Zaber)",
        },
        "camera": {
            "width": cam_width,
            "height": cam_height,
            "is_color": cam_is_color,
            "exposure_us": exposure,
            "gain_db": GAIN,
            "hardware_trigger": True,
            "trigger_line": "Line2",
        },
        "focus_positions": focus_summary,
        "grid_labels": GRID_LABELS,
        "timing": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_s": duration.total_seconds(),
            "duration_human": str(duration),
        },
        "title": title,
        "timestamp": run_ts,
        "total_images": len(all_images),
        "images": all_images,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults JSON saved to {out / 'results.json'}")

    # ── Save images.csv ─────────────────────────────────────────────────
    csv_path = out / "images.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "path", "focus_offset_mm", "focus_position_mm", "laplacian_variance",
            "target_shift_px", "tilt_x_deg", "tilt_y_deg",
            "cycle_index", "repetition", "position", "label",
            "commanded_x_deg", "commanded_y_deg",
        ])
        for img in all_images:
            writer.writerow([
                img["path"], img["focus_offset_mm"], img["focus_position_mm"],
                img["laplacian_variance"], img["target_shift_px"],
                img["tilt_x_deg"], img["tilt_y_deg"],
                img["cycle_index"], img["repetition"],
                img["position"], img["label"],
                img["commanded_x_deg"], img["commanded_y_deg"],
            ])
    print(f"Images CSV saved to {csv_path}")

    print(f"\nDone. {len(all_images)} images saved to {out}")
    print(f"Duration: {duration}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Focus-sweep SR data collection")
    parser.add_argument("--port", required=True,
                        help="Zaber serial port (e.g. /dev/ttyACM0)")
    parser.add_argument("--xpr-port", default=None,
                        help="XPR controller serial port (e.g. /dev/ttyACM1)")
    parser.add_argument("--calibration", required=True,
                        help="Path to pinhole experiment shifts.csv")
    parser.add_argument("--title", default="sr_sweep",
                        help="Experiment title (used in folder name)")
    parser.add_argument("--only-optimal", action="store_true",
                        help="Only capture at in-focus position with optimal shift (0.5px mono / 1.0px color)")
    args = parser.parse_args()
    run(zaber_port=args.port, xpr_port=args.xpr_port,
        calibration_csv=args.calibration, title=args.title,
        only_optimal=args.only_optimal)
