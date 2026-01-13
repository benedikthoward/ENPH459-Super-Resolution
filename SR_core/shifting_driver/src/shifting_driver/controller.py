"""
Optotune ICC-4C / XRP-20 Beam Shifter Controller

Provides a high-level interface for controlling the XRP-20 beam shifter
via the ICC-4C driver board.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from optoICC import connect

from .config import get_config


class WaveformShape(Enum):
    """Beam shifting waveform patterns."""
    MANHATTAN = 0  # Square pattern
    DIAMOND = 1    # Diamond pattern


@dataclass
class ShifterStatus:
    """Current status of the beam shifter."""
    is_running: bool
    frame_rate: int
    waveform: WaveformShape
    active_channel: int
    device_model: str


class BeamShifterController:
    """
    High-level controller for the Optotune XRP-20 beam shifter.
    
    The XRP-20 is controlled via the ICC-4C driver board, which connects
    via USB serial. The beam shifter uses two channels (0-1) for X/Y
    deflection control.
    
    Example:
        >>> ctrl = BeamShifterController()
        >>> ctrl.set_frame_rate(60)
        >>> ctrl.set_waveform(WaveformShape.MANHATTAN)
        >>> ctrl.start()
        >>> # ... capture images ...
        >>> ctrl.stop()
    """
    
    def __init__(self, port: Optional[str] = None, channel: Optional[int] = None):
        """
        Initialize connection to the ICC-4C controller.
        
        Args:
            port: Serial port for ICC-4C. If None, uses config.toml setting
                  or auto-detect if config is empty.
                  Examples: '/dev/cu.usbserial-xxx' (macOS),
                           '/dev/ttyUSB0' (Linux), 'COM3' (Windows)
            channel: Active channel for XRP control (0, 1, or 2).
                     If None, uses config.toml setting.
                     XRP devices use channel pairs, so channel 0 uses 0-1.
        """
        config = get_config()
        
        # Use config values if not explicitly provided
        if port is None:
            port = config.beam_shifter.port  # May be None for auto-detect
        if channel is None:
            channel = config.beam_shifter.channel
            
        self._port = port
        self._channel = channel
        self._default_frame_rate = config.beam_shifter.default_frame_rate
        self._default_waveform = config.beam_shifter.default_waveform
        self._board = None
        self._xpr_control = None
        self._is_connected = False
        
    def connect(self) -> None:
        """
        Establish connection to the ICC-4C board.
        
        Raises:
            ConnectionError: If unable to connect to the board.
        """
        try:
            if self._port:
                self._board = connect(port=self._port)
            else:
                self._board = connect()
            
            # Access XPR control system
            self._xpr_control = self._board.XPRControl
            
            # Set active channel (required to initialize device detection)
            self._xpr_control.SetActiveChannel(self._channel)
            
            self._is_connected = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to ICC-4C: {e}") from e
    
    def disconnect(self) -> None:
        """Stop shifting and close connection."""
        if self._is_connected:
            try:
                self.stop()
            except Exception:
                pass  # Best effort to stop
            self._board = None
            self._xpr_control = None
            self._is_connected = False
    
    def __enter__(self) -> "BeamShifterController":
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
    
    @property
    def is_connected(self) -> bool:
        """Check if controller is connected."""
        return self._is_connected
    
    def _ensure_connected(self) -> None:
        """Raise error if not connected."""
        if not self._is_connected:
            raise RuntimeError("Not connected to ICC-4C. Call connect() first.")
    
    def get_supported_frame_rates(self) -> list[float]:
        """
        Get list of frame rates supported by the connected XRP device.
        
        Returns:
            List of supported frame rates in Hz.
        """
        self._ensure_connected()
        return list(self._xpr_control.GetSupportedFramerates())
    
    def get_frame_rate(self) -> int:
        """Get current frame rate in Hz."""
        self._ensure_connected()
        return self._xpr_control.GetFrameRate()[0]
    
    def set_frame_rate(self, hz: int) -> None:
        """
        Set the beam shifting frame rate.
        
        Args:
            hz: Frame rate in Hz. Must be one of the supported rates
                (typically 50, 60, 90, 100, 120 Hz).
        
        Raises:
            ValueError: If frame rate is not supported.
        """
        self._ensure_connected()
        supported = self.get_supported_frame_rates()
        if hz not in supported:
            raise ValueError(
                f"Frame rate {hz} Hz not supported. "
                f"Supported rates: {supported}"
            )
        self._xpr_control.SetFrameRate(hz)
    
    def get_waveform(self) -> WaveformShape:
        """Get current waveform shape."""
        self._ensure_connected()
        shape_idx = self._xpr_control.GetWaveformShape()[0]
        return WaveformShape(shape_idx)
    
    def set_waveform(self, shape: WaveformShape) -> None:
        """
        Set the beam shifting waveform pattern.
        
        Args:
            shape: WaveformShape.MANHATTAN (square) or WaveformShape.DIAMOND
        """
        self._ensure_connected()
        self._xpr_control.SetWaveformShape(shape.value)
    
    def get_status(self) -> ShifterStatus:
        """
        Get comprehensive status of the beam shifter.
        
        Returns:
            ShifterStatus with current configuration.
        """
        self._ensure_connected()
        
        status_code = self._xpr_control.GetStatus()[0]
        is_running = status_code == 1
        
        device = self._xpr_control.GetActiveDevice()[0]
        device_names = {1: "XRP20", 8: "XRP33", 29: "XRP18", 31: "XRP26"}
        device_model = device_names.get(device, f"Unknown ({device})")
        
        return ShifterStatus(
            is_running=is_running,
            frame_rate=self.get_frame_rate(),
            waveform=self.get_waveform(),
            active_channel=self._xpr_control.GetActiveChannel()[0],
            device_model=device_model,
        )
    
    def start(self) -> None:
        """
        Start beam shifting with current configuration.
        
        The beam shifter will begin outputting the configured waveform
        at the set frame rate.
        """
        self._ensure_connected()
        self._xpr_control.SetLoad(1)  # 1 = Run
    
    def stop(self) -> None:
        """Stop beam shifting."""
        self._ensure_connected()
        self._xpr_control.SetLoad(0)  # 0 = Stop
    
    def is_running(self) -> bool:
        """Check if beam shifter is currently running."""
        self._ensure_connected()
        status = self._xpr_control.GetStatus()[0]
        return status == 1
