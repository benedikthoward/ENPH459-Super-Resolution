"""
Higher-level imaging API that coordinates the camera and XPR controller.

    from imaging import ImagingSystem

    with ImagingSystem() as sys:
        result = sys.capture_shifted([(0.1, 0.0), (-0.1, 0.0)])
        # result.images is a list of numpy arrays
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .camera import DahengCamera, TRIGGER_LINE2
from .xpr_controller import XPRController

log = logging.getLogger(__name__)


@dataclass
class CaptureResult:
    images: list[np.ndarray]
    shift_positions: list[tuple[float, float]]  # (angle_x_deg, angle_y_deg)
    exposure_us: float
    gain_db: float
    camera_resolution: tuple[int, int]
    settle_time_s: float
    timestamp: str


class ImagingSystem:
    """Coordinates DahengCamera and XPRController for shifted image capture.

    Parameters
    ----------
    hardware_trigger : bool
        If True, the XPR controller sends a GPIO pulse to trigger the camera
        instead of using a software trigger. Default False.
    trigger_line : int
        Which camera trigger input line to use for hardware triggering.
    """

    def __init__(self, hardware_trigger: bool = True, trigger_line: int = TRIGGER_LINE2,
                 xpr_port: str | None = None):
        log.info("Connecting to hardware...")
        self._hardware_trigger = hardware_trigger
        self._cam = DahengCamera(hardware_trigger=hardware_trigger, trigger_line=trigger_line)
        self._xpr = XPRController(port=xpr_port)
        if hardware_trigger:
            self._xpr.setup_trigger_output()
        log.info("Camera: %dx%d, color=%s, hw_trigger=%s",
                 self._cam.width, self._cam.height, self._cam.is_color, hardware_trigger)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._xpr.close()
        self._cam.close()

    @property
    def camera(self) -> DahengCamera:
        return self._cam

    @property
    def xpr(self) -> XPRController:
        return self._xpr

    def _capture_frame(self) -> np.ndarray:
        if self._hardware_trigger:
            self._xpr.send_trigger_pulse()
        return self._cam.capture_raw()

    def capture_shifted(
        self,
        positions: list[tuple[float, float]],
        settle_time_s: float = 0.005,
    ) -> CaptureResult:
        """Capture images at a list of (angle_x, angle_y) positions in degrees.

        Moves the XPR to each position, waits for settling, captures a frame,
        then returns the XPR to home.
        """
        images = []
        for x, y in positions:
            self._xpr.set_angles(x, y)
            time.sleep(settle_time_s)
            images.append(self._capture_frame())
        self._xpr.set_home()

        return CaptureResult(
            images=images,
            shift_positions=list(positions),
            exposure_us=self._cam.exposure,
            gain_db=self._cam.gain,
            camera_resolution=(self._cam.width, self._cam.height),
            settle_time_s=settle_time_s,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def capture_xpr_pattern(
        self,
        tilt_deg: float = 0.14391,
        settle_time_s: float = 0.005,
    ) -> CaptureResult:
        """Capture at the standard 4-position XPR shift pattern."""
        angles = XPRController.get_xpr_angles(tilt_deg)
        positions = [(float(row[0]), float(row[1])) for row in angles]
        return self.capture_shifted(positions, settle_time_s=settle_time_s)
