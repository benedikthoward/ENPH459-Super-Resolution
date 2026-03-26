"""
Stability calibration: sweep tilt angles x settling times, capture images
of a vertical knife edge, measure actual pixel shifts via edge fitting.

Place a sharp black/white boundary (left=dark, right=bright) in the FOV.
The script averages rows to get a 1D profile, finds the 50% crossing point
with linear interpolation for sub-pixel edge position, then compares
positions between XPR frames to get the shift.

Edit the parameters below, then run: python -m imaging.calibration.stability
"""

import json
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from imaging import XPRController, DahengCamera

# ── Parameters ──────────────────────────────────────────────────────────────
TILT_ANGLE_MIN = 0.02          # radians (GUI slider goes 0 to 0.5)
TILT_ANGLE_MAX = 0.50
TILT_ANGLE_STEPS = 15

SETTLING_TIME_MIN_MS = 2       # ms (optics transition takes ~1.5ms)
SETTLING_TIME_MAX_MS = 2000    # ms
SETTLING_TIME_STEPS = 10

NUM_REPEATS = 3
EXPOSURE = None                # µs, or None for auto
GAIN = 0                       # dB
OUTPUT_DIR = Path("./stability_data")
# ────────────────────────────────────────────────────────────────────────────

TILT_ANGLES = np.linspace(TILT_ANGLE_MIN, TILT_ANGLE_MAX, TILT_ANGLE_STEPS)
SETTLING_TIMES_MS = np.linspace(SETTLING_TIME_MIN_MS, SETTLING_TIME_MAX_MS, SETTLING_TIME_STEPS)


def find_edge_position(img: np.ndarray) -> float:
    """Find sub-pixel x-position of a vertical knife edge.
    Averages all rows into a 1D profile, finds the 50% crossing point."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    profile = gray.astype(np.float64).mean(axis=0)
    mid = (profile.min() + profile.max()) / 2.0
    for i in range(len(profile) - 1):
        if (profile[i] <= mid < profile[i + 1]) or (profile[i] >= mid > profile[i + 1]):
            t = (mid - profile[i]) / (profile[i + 1] - profile[i])
            return i + t
    return float(np.argmin(np.abs(profile - mid)))


def plot_results(results: dict, out: Path):
    tilt_angles = sorted(set(r["tilt_angle"] for r in results.values()))
    settle_times = sorted(set(r["settling_time_ms"] for r in results.values()))

    # -- Plot 1: shift magnitude vs tilt angle (one line per settling time) --
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for settle in settle_times:
        shifts = []
        for tilt in tilt_angles:
            key = f"tilt{tilt:.5f}_settle{settle:.1f}ms"
            if key in results:
                # average absolute shift across positions 1-3
                mean_dx = np.mean([abs(s["dx_mean"]) for s in results[key]["mean_shifts"]])
                shifts.append(mean_dx)
            else:
                shifts.append(np.nan)
        ax.plot(tilt_angles, shifts, "o-", label=f"{settle:.0f} ms", markersize=4)
    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="1 px target")
    ax.set_xlabel("Tilt angle (rad)")
    ax.set_ylabel("Mean |shift| (px)")
    ax.set_title("Shift vs tilt angle")
    ax.legend(fontsize=7, title="Settle time")
    ax.grid(True, alpha=0.3)

    # -- Plot 2: shift std (repeatability) vs settling time (one line per tilt) --
    ax = axes[1]
    for tilt in tilt_angles:
        stds = []
        for settle in settle_times:
            key = f"tilt{tilt:.5f}_settle{settle:.1f}ms"
            if key in results:
                mean_std = np.mean([s["dx_std"] for s in results[key]["mean_shifts"]])
                stds.append(mean_std)
            else:
                stds.append(np.nan)
        ax.plot(settle_times, stds, "o-", label=f"{tilt:.3f} rad", markersize=4)
    ax.set_xlabel("Settling time (ms)")
    ax.set_ylabel("Shift std across repeats (px)")
    ax.set_title("Repeatability vs settling time")
    ax.legend(fontsize=7, title="Tilt angle")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(out / "stability_plots.png"), dpi=150)
    plt.show()
    print(f"Plots saved to {out / 'stability_plots.png'}")


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
        print(f"Tilt angles: {len(TILT_ANGLES)} steps [{TILT_ANGLE_MIN:.4f} .. {TILT_ANGLE_MAX:.4f}] rad")
        print(f"Settling times: {len(SETTLING_TIMES_MS)} steps [{SETTLING_TIME_MIN_MS} .. {SETTLING_TIME_MAX_MS}] ms")
        print()

        results = {}
        total_combos = len(TILT_ANGLES) * len(SETTLING_TIMES_MS)
        combo = 0

        for ti, tilt in enumerate(TILT_ANGLES):
            angles = XPRController.get_xpr_angles(tilt)

            for si, settle in enumerate(SETTLING_TIMES_MS):
                combo += 1
                key = f"tilt{tilt:.5f}_settle{settle:.1f}ms"
                shifts_all_repeats = []

                for r in range(NUM_REPEATS):
                    xpr.set_angles(angles[0, 0], angles[0, 1])
                    time.sleep(settle / 1000.0)
                    ref_img = cam.capture_raw()
                    cv2.imwrite(str(out / f"{key}_rep{r:02d}_pos0.png"), ref_img)
                    ref_edge = find_edge_position(ref_img)

                    shifts = []
                    for p in range(1, 4):
                        xpr.set_angles(angles[p, 0], angles[p, 1])
                        time.sleep(settle / 1000.0)
                        img = cam.capture_raw()
                        cv2.imwrite(str(out / f"{key}_rep{r:02d}_pos{p}.png"), img)

                        edge = find_edge_position(img)
                        shifts.append({"pos": p, "dx": edge - ref_edge})

                    shifts_all_repeats.append(shifts)

                mean_shifts = []
                for p_idx in range(3):
                    dxs = [shifts_all_repeats[r][p_idx]["dx"] for r in range(NUM_REPEATS)]
                    mean_shifts.append({
                        "pos": p_idx + 1,
                        "dx_mean": float(np.mean(dxs)),
                        "dx_std": float(np.std(dxs)),
                    })

                results[key] = {
                    "tilt_angle": float(tilt),
                    "settling_time_ms": float(settle),
                    "mean_shifts": mean_shifts,
                    "all_repeats": shifts_all_repeats,
                }

                print(f"  [{combo}/{total_combos}] tilt={tilt:.5f}, settle={settle:.1f}ms:")
                for s in mean_shifts:
                    print(f"    pos{s['pos']}: dx={s['dx_mean']:+.3f} ± {s['dx_std']:.3f} px")

                xpr.set_home()

        (out / "shifts.json").write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {out / 'shifts.json'}")

    plot_results(results, out)
    print("Done.")


if __name__ == "__main__":
    run()
