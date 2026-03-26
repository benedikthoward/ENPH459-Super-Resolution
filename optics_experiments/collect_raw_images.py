"""
Collect raw images for offline super-resolution experiments.

Sweeps through several tilt angles and settle times to find the best
combination. The tilt angles are centered on the Daheng Mono nominal
value (0.14391 rad). For each (tilt_angle, settle_time) combination:

  1. Move XPR to the home (center) position.
  2. Wait settle_time for optics to settle.
  3. Capture and save the center image (no shift).
  4. For each of the 4 XPR shift positions:
       a. Command the XPR to the shifted angle.
       b. Wait settle_time for the optics to settle.
       c. Capture and save the raw image.
  5. Return the XPR to the home position before the next combination.

Camera exposure is set once at startup (auto or manual) and held fixed
for all captures — auto_exposure() does NOT re-run per image.

All images and a metadata JSON are saved to a timestamped subdirectory.

Edit the parameters below, then run:
    uv run python -m optics_experiments.collect_raw_images
"""

import json
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from imaging import XPRController, DahengCamera

# ── Parameters ──────────────────────────────────────────────────────────────
TILT_ANGLE_CENTER = 0.14391    # radians — Daheng Mono nominal 1px shift
TILT_ANGLE_SPREAD = 0.05      # radians — sweep ± this amount around center
TILT_ANGLE_STEPS = 20          # number of tilt angles to try

SETTLE_TIMES_MS = [2, 10, 50, 200, 1000]  # ms — biased toward low values

EXPOSURE = None                # µs, or None for auto (runs once at startup)
GAIN = 0                       # dB
OUTPUT_DIR = Path("./raw_capture_data")
# ────────────────────────────────────────────────────────────────────────────

TILT_ANGLES = np.linspace(
    TILT_ANGLE_CENTER - TILT_ANGLE_SPREAD,
    TILT_ANGLE_CENTER + TILT_ANGLE_SPREAD,
    TILT_ANGLE_STEPS,
)



def run():
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / run_ts
    out.mkdir(parents=True, exist_ok=True)

    total_combos = len(TILT_ANGLES) * len(SETTLE_TIMES_MS)

    with XPRController() as xpr, DahengCamera() as cam:
        cam.gain = GAIN
        if EXPOSURE is not None:
            cam.exposure = EXPOSURE
        else:
            cam.auto_exposure()

        params = {
            "tilt_angles": TILT_ANGLES.tolist(),
            "settle_times_ms": SETTLE_TIMES_MS,
            "exposure_us": cam.exposure,
            "gain_db": cam.gain,
            "camera": {"width": cam.width, "height": cam.height, "is_color": cam.is_color},
            "timestamp": run_ts,
        }

        print(f"Saving to {out}")
        print(f"Exposure: {cam.exposure} µs, Gain: {cam.gain} dB")
        print(f"Camera: {cam.width}x{cam.height}, color={cam.is_color}")
        print(f"Tilt angles: {TILT_ANGLES} rad")
        print(f"Settle times: {SETTLE_TIMES_MS} ms")
        print(f"Total combinations: {total_combos}")
        print()

        all_images = []
        combo = 0

        for tilt in TILT_ANGLES:
            for settle in SETTLE_TIMES_MS:
                combo += 1
                combo_label = f"tilt{tilt:.5f}_settle{settle:.1f}ms"
                combo_dir = out / combo_label
                combo_dir.mkdir(exist_ok=True)

                print(f"[{combo}/{total_combos}] tilt = {tilt:.5f} rad, settle = {settle:.1f} ms")

                # Capture center (home) image
                xpr.set_home()
                time.sleep(settle / 1000.0)
                center_img = cam.capture_raw()
                center_name = f"{combo_label}/center.png"
                cv2.imwrite(str(out / center_name), center_img)
                all_images.append(center_name)
                print(f"  Captured center (home)")

                # Capture 4 shifted images
                angles = XPRController.get_xpr_angles(tilt)
                for p in range(4):
                    xpr.set_angles(angles[p, 0], angles[p, 1])
                    time.sleep(settle / 1000.0)
                    img = cam.capture_raw()
                    shift_name = f"{combo_label}/shift_{p}.png"
                    cv2.imwrite(str(out / shift_name), img)
                    all_images.append(shift_name)
                    print(f"  Captured shift_{p} (angle_x={angles[p, 0]:+.5f}, angle_y={angles[p, 1]:+.5f})")

                # Return to home before next combo
                xpr.set_home()

        params["images"] = all_images
        (out / "params.json").write_text(json.dumps(params, indent=2))

    print(f"\nDone. {len(all_images)} images saved to {out}")


if __name__ == "__main__":
    run()
