import cv2
import time
import numpy as np
import gxipy as gx

# gxipy trigger source values (from gxidef.py)
TRIGGER_SOFTWARE = 0
TRIGGER_LINE0 = 1   # opto-isolated input
TRIGGER_LINE2 = 3   # GPIO (3.3V logic)
TRIGGER_LINE3 = 4   # GPIO (3.3V logic)


class DahengCamera:
    def __init__(self, device_index: int = 0, hardware_trigger: bool = False,
                 trigger_line: int = TRIGGER_LINE2):
        self._dm = gx.DeviceManager()
        dev_num, dev_info_list = self._dm.update_device_list()
        if dev_num == 0:
            raise RuntimeError("No Daheng camera detected")

        sn = dev_info_list[device_index].get("sn")
        self._cam = self._dm.open_device_by_sn(sn)
        self._cam.data_stream[0].StreamBufferHandlingMode.set(3)
        self._cam.TriggerMode.set(1)
        self._cam.Gain.set(0)

        self._hardware_trigger = hardware_trigger
        if hardware_trigger:
            self._cam.TriggerSource.set(trigger_line)
            self._cam.TriggerActivation.set(1)  # rising edge
        else:
            self._cam.TriggerSource.set(TRIGGER_SOFTWARE)

        self._cam.stream_on()

        try:
            self._is_color = self._cam.PixelColorFilter.get() != gx.GxPixelColorFilterEntry.NONE
        except Exception:
            self._is_color = False

        self._width = self._cam.Width.get()
        self._height = self._cam.Height.get()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._cam.stream_off()
        self._cam.close_device()

    def _get_timeout_ms(self) -> int:
        """Timeout for get_image: exposure time + 2s margin."""
        return int(self._cam.ExposureTime.get() / 1000) + 2000

    def capture_raw(self) -> np.ndarray:
        if not self._hardware_trigger:
            self._cam.TriggerSoftware.send_command()
        raw = self._cam.data_stream[0].get_image(timeout=self._get_timeout_ms())
        if raw is None:
            raise RuntimeError("Failed to capture image")
        return raw.get_numpy_array()

    def capture_rgb(self) -> np.ndarray:
        if not self._hardware_trigger:
            self._cam.TriggerSoftware.send_command()
        raw = self._cam.data_stream[0].get_image(timeout=self._get_timeout_ms())
        if raw is None:
            raise RuntimeError("Failed to capture image")
        if self._is_color:
            return raw.convert("RGB", convert_type=0).get_numpy_array()
        else:
            return cv2.cvtColor(raw.get_numpy_array(), cv2.COLOR_GRAY2RGB)

    @property
    def exposure(self) -> float:
        return self._cam.ExposureTime.get()

    @exposure.setter
    def exposure(self, value: float):
        self._cam.ExposureTime.set(value)

    @property
    def gain(self) -> float:
        return self._cam.Gain.get()

    @gain.setter
    def gain(self, value: float):
        self._cam.Gain.set(value)

    def auto_exposure(self) -> float:
        self._cam.ExposureAuto.set(2)
        time.sleep(1)
        value = self._cam.ExposureTime.get()
        self._cam.ExposureAuto.set(0)
        return value

    def auto_white_balance(self):
        if self._is_color and self._cam.BalanceWhiteAuto.is_writable():
            self._cam.BalanceWhiteAuto.set(2)

    @property
    def is_color(self) -> bool:
        return self._is_color

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height
