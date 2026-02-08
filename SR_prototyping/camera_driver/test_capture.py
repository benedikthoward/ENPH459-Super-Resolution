#!/usr/bin/env python3
"""
Quick camera test — discover, connect, capture, and save a single image.

Usage:
    python -m camera_driver.test_capture
    python -m camera_driver.test_capture --output my_photo.tiff
"""

import argparse
import logging
import sys
from pathlib import Path

from camera_driver import Camera


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a test image from a Vimba camera")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="test_capture.tiff",
        help="Output file path (default: test_capture.tiff)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s — %(message)s",
    )

    # ---- 1. Discover cameras ------------------------------------------------
    print("Searching for cameras …")
    cameras = Camera.list_cameras()

    if not cameras:
        print("No cameras found. Make sure the camera is connected and the Vimba drivers are installed.")
        sys.exit(1)

    print(f"Found {len(cameras)} camera(s):")
    for i, info in enumerate(cameras):
        print(f"  [{i}] {info.name}  (model={info.model}, serial={info.serial}, id={info.camera_id})")

    # ---- 2. Connect to the first camera -------------------------------------
    print("\nConnecting to the first camera …")
    with Camera() as cam:
        info = cam.info
        print(f"Connected: {info.name} (serial {info.serial})")

        # ---- 3. Capture an image --------------------------------------------
        print("Capturing frame …")
        image = cam.capture()
        print(f"Frame captured — shape={image.shape}, dtype={image.dtype}")

        # ---- 4. Save --------------------------------------------------------
        out_path = cam.save(image, args.output)
        print(f"Image saved to {out_path.resolve()}")

    print("Done ✓")


if __name__ == "__main__":
    main()
