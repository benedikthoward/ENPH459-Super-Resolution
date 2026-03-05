"""
Stability calibration: sweep tilt angles x settling times, capture raw images
of a knife edge, and measure actual pixel shifts via phase correlation.

Edit the parameters below, then run: python -m imaging.calibration.stability
"""

import json
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from imaging import XPRController, DahengCamera

# ── Parameters ──────────────────────────────────────────────────────────────
TILT_ANGLE_MIN = 0.02          # radians
TILT_ANGLE_MAX = 0.30
TILT_ANGLE_STEPS = 15

SETTLING_TIME_MIN_MS = 1       # milliseconds
SETTLING_TIME_MAX_MS = 30
SETTLING_TIME_STEPS = 10

NUM_REPEATS = 3                # full 4-position cycles per (tilt, settle) combo
EXPOSURE = None                # µs, or None for auto
GAIN = 0                       # dB
OUTPUT_DIR = Path("./stability_data")
# ────────────────────────────────────────────────────────────────────────────

TILT_ANGLES = np.linspace(TILT_ANGLE_MIN, TILT_ANGLE_MAX, TILT_ANGLE_STEPS)
SETTLING_TIMES_MS = np.linspace(SETTLING_TIME_MIN_MS, SETTLING_TIME_MAX_MS, SETTLING_TIME_STEPS)


def measure_shift(ref: np.ndarray, img: np.ndarray) -> tuple[float, float]:
    """Sub-pixel (dx, dy) shift of img relative to ref using phase correlation."""
    ref_f = ref.astype(np.float64)
    img_f = img.astype(np.float64)
    (dx, dy), response = cv2.phaseCorrelate(ref_f, img_f)
    return dx, dy


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
            "settling_times_ms": SETTLING_TIMES_MS.tolist(),
            "num_repeats": NUM_REPEATS,
            "exposure_us": cam.exposure,
            "gain_db": cam.gain,
            "camera": {"width": cam.width, "height": cam.height, "is_color": cam.is_color},
            "timestamp": run_ts,
        }
        (out / "params.json").write_text(json.dumps(params, indent=2))

        print(f"Saving to {out}")
        print(f"Exposure: {cam.exposure} µs, Gain: {cam.gain} dB")
        print(f"Camera: {cam.width}x{cam.height}, color={cam.is_color}")
        print(f"Tilt angles: {len(TILT_ANGLES)} steps [{TILT_ANGLE_MIN:.4f} .. {TILT_ANGLE_MAX:.4f}]")
        print(f"Settling times: {len(SETTLING_TIMES_MS)} steps [{SETTLING_TIME_MIN_MS} .. {SETTLING_TIME_MAX_MS}] ms")
        print()

        # results[tilt_idx][settle_idx] = list of (dx, dy) per position 1-3, per repeat
        results = {}

        for ti, tilt in enumerate(TILT_ANGLES):
            angles = XPRController.get_xpr_angles(tilt)

            for si, settle in enumerate(SETTLING_TIMES_MS):
                key = f"tilt{tilt:.5f}_settle{settle:.1f}ms"
                shifts_all_repeats = []

                for r in range(NUM_REPEATS):
                    # capture position 0 as reference
                    xpr.set_angles(angles[0, 0], angles[0, 1])
                    time.sleep(settle / 1000.0)
                    ref_img = cam.capture_raw()
                    fname = f"{key}_rep{r:02d}_pos0.png"
                    cv2.imwrite(str(out / fname), ref_img)

                    # convert to grayscale for phase correlation
                    if ref_img.ndim == 3:
                        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                    else:
                        ref_gray = ref_img

                    shifts = []
                    for p in range(1, 4):
                        xpr.set_angles(angles[p, 0], angles[p, 1])
                        time.sleep(settle / 1000.0)
                        img = cam.capture_raw()
                        fname = f"{key}_rep{r:02d}_pos{p}.png"
                        cv2.imwrite(str(out / fname), img)

                        if img.ndim == 3:
                            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        else:
                            img_gray = img

                        dx, dy = measure_shift(ref_gray, img_gray)
                        shifts.append({"pos": p, "dx": dx, "dy": dy})

                    shifts_all_repeats.append(shifts)

                # compute mean shifts across repeats
                mean_shifts = []
                for p_idx in range(3):  # positions 1, 2, 3
                    dxs = [shifts_all_repeats[r][p_idx]["dx"] for r in range(NUM_REPEATS)]
                    dys = [shifts_all_repeats[r][p_idx]["dy"] for r in range(NUM_REPEATS)]
                    mean_shifts.append({
                        "pos": p_idx + 1,
                        "dx_mean": float(np.mean(dxs)),
                        "dy_mean": float(np.mean(dys)),
                        "dx_std": float(np.std(dxs)),
                        "dy_std": float(np.std(dys)),
                    })

                results[key] = {
                    "tilt_angle": float(tilt),
                    "settling_time_ms": float(settle),
                    "mean_shifts": mean_shifts,
                    "all_repeats": shifts_all_repeats,
                }

                # print summary for this combo
                print(f"  tilt={tilt:.5f}, settle={settle:.1f}ms:")
                for s in mean_shifts:
                    print(f"    pos{s['pos']}: dx={s['dx_mean']:+.3f}±{s['dx_std']:.3f}  "
                          f"dy={s['dy_mean']:+.3f}±{s['dy_std']:.3f} px")

                xpr.set_home()

        (out / "shifts.json").write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {out / 'shifts.json'}")

    print("Done.")


if __name__ == "__main__":
    run()
