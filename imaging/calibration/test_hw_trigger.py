"""
Quick test: capture a few images using hardware triggering.

    uv run python -m imaging.calibration.test_hw_trigger
"""

import cv2
from pathlib import Path
from imaging import ImagingSystem

out = Path("./hw_trigger_test")
out.mkdir(exist_ok=True)

with ImagingSystem() as sys:
    print(f"Camera: {sys.camera.width}x{sys.camera.height}")
    print(f"Exposure: {sys.camera.exposure} µs, Gain: {sys.camera.gain} dB")
    print(f"Hardware trigger: on")
    print()

    # single frames at home position
    for i in range(5):
        img = sys._capture_frame()
        fname = out / f"frame_{i}.png"
        cv2.imwrite(str(fname), img)
        print(f"  frame {i}: {img.shape}, mean={img.mean():.1f} -> {fname}")

    # XPR 4-position pattern
    print("\nCapturing XPR pattern...")
    result = sys.capture_xpr_pattern()
    for i, img in enumerate(result.images):
        fname = out / f"xpr_pos{i}.png"
        cv2.imwrite(str(fname), img)
        print(f"  pos {i}: {img.shape}, mean={img.mean():.1f} -> {fname}")

print(f"\nDone. Images in {out}/")
