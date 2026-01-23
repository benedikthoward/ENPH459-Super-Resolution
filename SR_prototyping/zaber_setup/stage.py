"""
Zaber Motion Stage Controller

Provides control over Zaber linear/rotary stages for sample positioning.
"""

from dataclasses import dataclass
from typing import Optional

from zaber_motion import Units
from zaber_motion.ascii import Connection, Device, Axis

from .config import get_config


@dataclass
class StagePosition:
    """Current position of the stage."""
    position_mm: float
    is_homed: bool
    is_busy: bool


class StageController:
    """
    Controller for Zaber motion stages.
    
    Used for precise sample positioning during super-resolution imaging.
    
    Example:
        >>> stage = StageController()
        >>> stage.connect()
        >>> stage.home()
        >>> stage.move_absolute(10.0)  # Move to 10mm
        >>> stage.move_relative(5.0)   # Move additional 5mm
        >>> stage.disconnect()
    
    Or with context manager:
        >>> with StageController() as stage:
        ...     stage.home()
        ...     stage.move_absolute(10.0)
    """
    
    def __init__(self, port: Optional[str] = None, axis_number: int = 1):
        """
        Initialize the stage controller.
        
        Args:
            port: Serial port for Zaber controller. If None, uses config.toml setting.
                  Examples: '/dev/tty.usbserial-xxx' (macOS),
                           '/dev/ttyUSB0' (Linux), 'COM3' (Windows)
            axis_number: Axis number to control (default: 1)
        """
        config = get_config()
        
        if port is None:
            port = config.stage.port if config.stage.port else None
        
        self._port = port
        self._axis_number = axis_number
        self._connection: Optional[Connection] = None
        self._device: Optional[Device] = None
        self._axis: Optional[Axis] = None
        self._is_connected = False
    
    def connect(self) -> None:
        """
        Connect to the Zaber stage.
        
        Raises:
            ConnectionError: If unable to connect or no devices found.
            ValueError: If no port specified and none in config.
        """
        if self._port is None:
            raise ValueError(
                "No serial port specified. Provide port argument or set in config.toml"
            )
        
        try:
            self._connection = Connection.open_serial_port(self._port)
            self._connection.enable_alerts()
            
            device_list = self._connection.detect_devices()
            if not device_list:
                raise ConnectionError("No Zaber devices found")
            
            self._device = device_list[0]
            self._axis = self._device.get_axis(self._axis_number)
            self._is_connected = True
            
        except Exception as e:
            self._cleanup()
            raise ConnectionError(f"Failed to connect to Zaber stage: {e}") from e
    
    def disconnect(self) -> None:
        """Disconnect from the stage."""
        self._cleanup()
    
    def _cleanup(self) -> None:
        """Clean up connection resources."""
        self._axis = None
        self._device = None
        if self._connection:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None
        self._is_connected = False
    
    def __enter__(self) -> "StageController":
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to stage."""
        return self._is_connected
    
    def _ensure_connected(self) -> None:
        """Raise error if not connected."""
        if not self._is_connected or not self._axis:
            raise RuntimeError("Not connected to Zaber stage. Call connect() first.")
    
    def home(self, wait: bool = True) -> None:
        """
        Home the stage axis.
        
        Args:
            wait: If True, block until homing is complete.
        """
        self._ensure_connected()
        self._axis.home(wait_until_idle=wait)
    
    def is_homed(self) -> bool:
        """Check if the axis has been homed."""
        self._ensure_connected()
        return self._axis.is_homed()
    
    def move_absolute(self, position_mm: float, wait: bool = True) -> None:
        """
        Move to an absolute position.
        
        Args:
            position_mm: Target position in millimeters.
            wait: If True, block until move is complete.
        """
        self._ensure_connected()
        self._axis.move_absolute(position_mm, Units.LENGTH_MILLIMETRES, wait_until_idle=wait)
    
    def move_relative(self, distance_mm: float, wait: bool = True) -> None:
        """
        Move relative to current position.
        
        Args:
            distance_mm: Distance to move in millimeters (positive or negative).
            wait: If True, block until move is complete.
        """
        self._ensure_connected()
        self._axis.move_relative(distance_mm, Units.LENGTH_MILLIMETRES, wait_until_idle=wait)
    
    def get_position(self) -> float:
        """
        Get current position in millimeters.
        
        Returns:
            Current position in mm.
        """
        self._ensure_connected()
        return self._axis.get_position(Units.LENGTH_MILLIMETRES)
    
    def get_status(self) -> StagePosition:
        """
        Get comprehensive status of the stage.
        
        Returns:
            StagePosition with current state.
        """
        self._ensure_connected()
        return StagePosition(
            position_mm=self.get_position(),
            is_homed=self.is_homed(),
            is_busy=self._axis.is_busy(),
        )
    
    def stop(self) -> None:
        """Emergency stop - halt all motion immediately."""
        self._ensure_connected()
        self._axis.stop()
    
    def wait_until_idle(self) -> None:
        """Block until the axis has finished moving."""
        self._ensure_connected()
        self._axis.wait_until_idle()
