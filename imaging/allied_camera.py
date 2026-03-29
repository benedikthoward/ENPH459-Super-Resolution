"""Allied Vision camera wrapper using VmbPy SDK."""

import os
import time
import threading
import numpy as np
import vmbpy

# Ensure transport layer is found
if "/opt/VimbaX_2026-1/cti" not in os.environ.get("GENICAM_GENTL64_PATH", ""):
    os.environ["GENICAM_GENTL64_PATH"] = os.environ.get("GENICAM_GENTL64_PATH", "") + ":/opt/VimbaX_2026-1/cti"


class AlliedCamera:
    def __init__(self, camera_id: str = None, exposure_us: float = 5000):
        self._vmb = vmbpy.VmbSystem.get_instance()
        self._vmb.__enter__()

        if camera_id:
            self._cam = self._vmb.get_camera_by_id(camera_id)
        else:
            cams = self._vmb.get_all_cameras()
            # skip simulators
            real = [c for c in cams if "Simulator" not in c.get_name()]
            if not real:
                raise RuntimeError("No Allied Vision camera found")
            self._cam = real[0]

        self._cam.__enter__()
        self._cam.set_pixel_format(vmbpy.PixelFormat.Mono8)
        self._cam.ExposureTime.set(exposure_us)
        try:
            self._cam.DeviceLinkThroughputLimit.set(450000000)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        try:
            self._cam.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self._vmb.__exit__(None, None, None)
        except Exception:
            pass

    @property
    def width(self) -> int:
        return self._cam.Width.get()

    @width.setter
    def width(self, value: int):
        self._cam.Width.set(value)

    @property
    def height(self) -> int:
        return self._cam.Height.get()

    @height.setter
    def height(self, value: int):
        self._cam.Height.set(value)

    @property
    def exposure(self) -> float:
        return self._cam.ExposureTime.get()

    @exposure.setter
    def exposure(self, value: float):
        self._cam.ExposureTime.set(value)

    @property
    def max_fps(self) -> float:
        return self._cam.AcquisitionFrameRate.get_range()[1]

    @property
    def model(self) -> str:
        return self._cam.get_model()

    def capture(self, timeout_ms: int = 10000) -> np.ndarray:
        """Single frame capture. Returns H×W uint8."""
        frame = self._cam.get_frame(timeout_ms=timeout_ms)
        return frame.as_numpy_ndarray().squeeze()

    def stream_burst(self, num_frames: int, buffer_count: int = 20,
                     timeout_s: float = 60) -> tuple[list[np.ndarray], list[float]]:
        """Stream num_frames at max FPS. Returns (images, timestamps_ms)."""
        frames = []
        timestamps = []
        lock = threading.Lock()
        done = threading.Event()
        t0 = [None]

        def handler(cam, stream, frame):
            if frame.get_status() == vmbpy.FrameStatus.Complete:
                now = time.perf_counter()
                with lock:
                    if len(frames) < num_frames:
                        if t0[0] is None:
                            t0[0] = now
                        frames.append(frame.as_numpy_ndarray().squeeze().copy())
                        timestamps.append((now - t0[0]) * 1000)
                        if len(frames) >= num_frames:
                            done.set()
            cam.queue_frame(frame)

        self._cam.start_streaming(handler, buffer_count=buffer_count)
        done.wait(timeout=timeout_s)
        self._cam.stop_streaming()
        return frames, timestamps
