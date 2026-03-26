import numpy as np
import optoICC
from optoControllerToolbox.SmartFilter import SmartFilters


class XPRController:
    def __init__(self):
        self._icc = optoICC.connect()
        self._icc.reset(force=True)
        self._icc.go_pro()

        for i in range(2):
            ch = self._icc.channel[i]
            ch.StaticInput.SetAsInput()
            ch.InputConditioning.SetGain(1.0)
            ch.SetControlMode(optoICC.UnitType.UNITLESS)

        self._si = [self._icc.channel[i].StaticInput for i in range(2)]
        self._si[0].SetValue(0)
        self._si[1].SetValue(0)

        self._smart_filters = SmartFilters(self._icc)
        self._smart_filters.transition_time = 1.5e-3
        self._smart_filters.channels = [0, 1]
        self._smart_filters.configure_filters()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.set_home()

    def set_angles(self, angle_x: float, angle_y: float):
        self._icc.set_value(
            [self._si[0].value, self._si[1].value],
            [float(angle_x), float(angle_y)]
        )

    def set_home(self):
        self.set_angles(0, 0)

    @property
    def transition_time(self) -> float:
        return self._smart_filters.transition_time

    @transition_time.setter
    def transition_time(self, value: float):
        self._smart_filters.transition_time = value
        self._smart_filters.configure_filters()

    @property
    def input_gain(self) -> float:
        return self._icc.channel[0].InputConditioning.GetGain()

    @input_gain.setter
    def input_gain(self, value: float):
        for i in range(2):
            self._icc.channel[i].InputConditioning.SetGain(value)

    @property
    def connected_devices(self) -> list:
        return [
            optoICC.DeviceModel(self._icc.MiscFeatures.GetDeviceType(i))
            for i in range(4)
        ]

    @staticmethod
    def get_xpr_angles(tilt_angle: float) -> np.ndarray:
        px_shifts = np.array([[-1, 1], [-1, -1], [1, -1], [1, 1]])
        return tilt_angle * px_shifts
