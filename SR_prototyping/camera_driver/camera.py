"""
Allied Vision (Vimba) Camera Controller
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from vimba import (
    Vimba,
    VimbaCameraError,
    VimbaFeatureError,
    VimbaTimeout,
    PixelFormat,
    intersect_pixel_formats,
    OPENCV_PIXEL_FORMATS,
    COLOR_PIXEL_FORMATS,
    MONO_PIXEL_FORMATS,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CameraInfo:
    """Read-only snapshot of camera identification details."""
    name: str
    model: str
    camera_id: str
    serial: str
    interface_id: str


# ---------------------------------------------------------------------------
# Camera class
# ---------------------------------------------------------------------------

class Camera:
    """
    Controller for Allied Vision cameras via the Vimba SDK.

    Args:
        camera_id:  Vimba camera ID string.  When *None* the first
                    discovered camera is used.
        exposure_us:  Fixed exposure time in microseconds.  Overrides
                      *auto_exposure* when set.
        auto_exposure:  Enable continuous auto-exposure (default True).
        auto_white_balance:  Enable continuous auto white-balance (default True).
        prefer_color:  When choosing a pixel format, prefer colour over
                       monochrome (default True).
    """

    def __init__(
        self,
        camera_id: Optional[str] = None,
        exposure_us: Optional[float] = None,
        auto_exposure: bool = True,
        auto_white_balance: bool = True,
        prefer_color: bool = True,
    ) -> None:
        self._camera_id = camera_id
        self._exposure_us = exposure_us
        self._auto_exposure = auto_exposure
        self._auto_white_balance = auto_white_balance
        self._prefer_color = prefer_color

        # Runtime state — populated by connect()
        self._vimba_instance: Optional[Vimba] = None
        self._cam = None  # vimba.Camera once opened
        self._is_connected: bool = False

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def list_cameras() -> list[CameraInfo]:
        """
        Discover all Vimba-accessible cameras.

        Returns:
            A list of :class:`CameraInfo` objects, one per camera found.
        """
        infos: list[CameraInfo] = []
        with Vimba.get_instance() as vimba:
            for cam in vimba.get_all_cameras():
                infos.append(
                    CameraInfo(
                        name=cam.get_name(),
                        model=cam.get_model(),
                        camera_id=cam.get_id(),
                        serial=cam.get_serial(),
                        interface_id=cam.get_interface_id(),
                    )
                )
        return infos

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Open the Vimba runtime, find the camera, and apply configuration.

        Raises:
            RuntimeError: If no cameras are found or the requested camera
                          ID does not exist.
        """
        if self._is_connected:
            log.warning("Camera is already connected")
            return

        try:
            # 1. Enter Vimba runtime
            self._vimba_instance = Vimba.get_instance()
            self._vimba_instance.__enter__()

            # 2. Locate camera
            if self._camera_id:
                try:
                    self._cam = self._vimba_instance.get_camera_by_id(self._camera_id)
                except VimbaCameraError as exc:
                    raise RuntimeError(
                        f"Camera '{self._camera_id}' not accessible"
                    ) from exc
            else:
                cams = self._vimba_instance.get_all_cameras()
                if not cams:
                    raise RuntimeError("No Vimba cameras found")
                self._cam = cams[0]

            # 3. Open camera context
            self._cam.__enter__()

            # 4. Apply configuration
            self._configure()

            self._is_connected = True
            info = self.info
            log.info("Connected to %s (serial %s)", info.name, info.serial)

        except Exception:
            # Roll back partial setup
            self._cleanup()
            raise

    def disconnect(self) -> None:
        """Close the camera and shut down the Vimba runtime."""
        self._cleanup()
        log.info("Camera disconnected")

    def _cleanup(self) -> None:
        """Release resources in reverse order of acquisition."""
        if self._cam is not None:
            try:
                self._cam.__exit__(None, None, None)
            except Exception:
                pass
            self._cam = None

        if self._vimba_instance is not None:
            try:
                self._vimba_instance.__exit__(None, None, None)
            except Exception:
                pass
            self._vimba_instance = None

        self._is_connected = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Camera":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Whether the camera is currently open and usable."""
        return self._is_connected

    @property
    def info(self) -> CameraInfo:
        """
        Identification details of the connected camera.

        Raises:
            RuntimeError: If not connected.
        """
        self._ensure_connected()
        return CameraInfo(
            name=self._cam.get_name(),
            model=self._cam.get_model(),
            camera_id=self._cam.get_id(),
            serial=self._cam.get_serial(),
            interface_id=self._cam.get_interface_id(),
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _configure(self) -> None:
        """Apply exposure, white-balance, GigE, and pixel-format settings."""
        self._configure_exposure()
        self._configure_white_balance()
        self._configure_gige()
        self._configure_pixel_format()

    def _configure_exposure(self) -> None:
        """Set exposure — either a fixed value or continuous auto."""
        try:
            if self._exposure_us is not None:
                self._cam.ExposureAuto.set("Off")
                self._cam.ExposureTime.set(self._exposure_us)
                log.debug("Exposure set to %.0f µs", self._exposure_us)
            elif self._auto_exposure:
                self._cam.ExposureAuto.set("Continuous")
                log.debug("Auto-exposure enabled")
        except (AttributeError, VimbaFeatureError) as exc:
            log.debug("Could not configure exposure: %s", exc)

    def _configure_white_balance(self) -> None:
        """Enable or skip continuous auto white-balance."""
        try:
            if self._auto_white_balance:
                self._cam.BalanceWhiteAuto.set("Continuous")
                log.debug("Auto white-balance enabled")
        except (AttributeError, VimbaFeatureError) as exc:
            log.debug("Could not configure white balance: %s", exc)

    def _configure_gige(self) -> None:
        """Adjust GigE packet size if the camera supports it."""
        try:
            self._cam.GVSPAdjustPacketSize.run()
            while not self._cam.GVSPAdjustPacketSize.is_done():
                pass
            log.debug("GigE packet size adjusted")
        except (AttributeError, VimbaFeatureError):
            pass  # Not a GigE camera — silently skip

    def _configure_pixel_format(self) -> None:
        """Pick an OpenCV-compatible pixel format (colour preferred)."""
        cv_fmts = intersect_pixel_formats(
            self._cam.get_pixel_formats(), OPENCV_PIXEL_FORMATS
        )
        color_fmts = intersect_pixel_formats(cv_fmts, COLOR_PIXEL_FORMATS)
        mono_fmts = intersect_pixel_formats(cv_fmts, MONO_PIXEL_FORMATS)

        if self._prefer_color and color_fmts:
            self._cam.set_pixel_format(color_fmts[0])
        elif mono_fmts:
            self._cam.set_pixel_format(mono_fmts[0])
        elif color_fmts:
            self._cam.set_pixel_format(color_fmts[0])
        else:
            raise RuntimeError(
                "Camera does not support any OpenCV-compatible pixel format"
            )
        log.debug("Pixel format: %s", self._cam.get_pixel_format())

    def set_exposure(self, exposure_us: float) -> None:
        """
        Change the exposure time on the fly (disables auto-exposure).

        Args:
            exposure_us: Exposure time in microseconds.
        """
        self._ensure_connected()
        try:
            self._cam.ExposureAuto.set("Off")
            self._cam.ExposureTime.set(exposure_us)
            self._exposure_us = exposure_us
            self._auto_exposure = False
            log.debug("Exposure updated to %.0f µs", exposure_us)
        except (AttributeError, VimbaFeatureError) as exc:
            raise RuntimeError(f"Failed to set exposure: {exc}") from exc

    def set_auto_exposure(self) -> None:
        """Switch back to continuous auto-exposure."""
        self._ensure_connected()
        try:
            self._cam.ExposureAuto.set("Continuous")
            self._exposure_us = None
            self._auto_exposure = True
            log.debug("Auto-exposure re-enabled")
        except (AttributeError, VimbaFeatureError) as exc:
            raise RuntimeError(f"Failed to enable auto-exposure: {exc}") from exc

    # ------------------------------------------------------------------
    # Image capture
    # ------------------------------------------------------------------

    def capture(self, timeout_ms: int = 5000) -> np.ndarray:
        """
        Capture a single frame and return it as a NumPy (OpenCV) array.

        Args:
            timeout_ms: Maximum time to wait for a frame in milliseconds.

        Returns:
            A NumPy array in BGR format (OpenCV convention).

        Raises:
            RuntimeError: If not connected or the capture times out.
        """
        self._ensure_connected()
        try:
            frame = self._cam.get_frame(timeout_ms=timeout_ms)
            image = frame.as_opencv_image()
            log.debug("Captured frame (%dx%d)", image.shape[1], image.shape[0])
            return image
        except VimbaTimeout as exc:
            raise RuntimeError(
                f"Frame capture timed out after {timeout_ms} ms"
            ) from exc

    def capture_mono(self, timeout_ms: int = 5000) -> np.ndarray:
        """
        Capture a single frame and convert it to 8-bit monochrome.

        Args:
            timeout_ms: Maximum time to wait for a frame in milliseconds.

        Returns:
            A 2-D NumPy array (H×W, dtype uint8).
        """
        self._ensure_connected()
        try:
            frame = self._cam.get_frame(timeout_ms=timeout_ms)
            frame.convert_pixel_format(PixelFormat.Mono8)
            image = frame.as_opencv_image()
            # as_opencv_image may still return (H, W, 1); squeeze to 2-D
            return image.squeeze()
        except VimbaTimeout as exc:
            raise RuntimeError(
                f"Frame capture timed out after {timeout_ms} ms"
            ) from exc

    # ------------------------------------------------------------------
    # Convenience I/O
    # ------------------------------------------------------------------

    @staticmethod
    def save(image: np.ndarray, path: str | Path) -> Path:
        """
        Write an image array to disk via OpenCV.

        Supports common formats inferred from the file extension
        (.tiff, .png, .jpg, .bmp, …).

        Args:
            image: NumPy image array (as returned by :meth:`capture`).
            path:  Destination file path.

        Returns:
            The resolved :class:`~pathlib.Path` that was written.

        Raises:
            IOError: If OpenCV cannot write the file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise IOError(f"Failed to write image to {path}")
        log.info("Image saved to %s", path)
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        if not self._is_connected or self._cam is None:
            raise RuntimeError(
                "Camera is not connected. Call connect() first."
            )
