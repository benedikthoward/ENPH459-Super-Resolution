"""
Physical scale and unit conversion for scene representation.

Maintains the mapping between pixel coordinates and physical (mm) coordinates
throughout the simulation pipeline.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class PhysicalExtent:
    """
    Defines the physical extent of a scene or image region.
    
    Attributes:
        width_mm: Physical width in millimeters
        height_mm: Physical height in millimeters
        center_x_mm: X coordinate of center in mm (default 0)
        center_y_mm: Y coordinate of center in mm (default 0)
    """
    width_mm: float
    height_mm: float
    center_x_mm: float = 0.0
    center_y_mm: float = 0.0
    
    @property
    def left_mm(self) -> float:
        return self.center_x_mm - self.width_mm / 2
    
    @property
    def right_mm(self) -> float:
        return self.center_x_mm + self.width_mm / 2
    
    @property
    def top_mm(self) -> float:
        return self.center_y_mm - self.height_mm / 2
    
    @property
    def bottom_mm(self) -> float:
        return self.center_y_mm + self.height_mm / 2
    
    @property
    def bounds_mm(self) -> Tuple[float, float, float, float]:
        """Returns (left, right, top, bottom) in mm."""
        return (self.left_mm, self.right_mm, self.top_mm, self.bottom_mm)


@dataclass
class SceneUnits:
    """
    Manages coordinate transformations between pixel and physical space.
    
    The internal oversampled representation uses a higher resolution than the
    final sensor output to enable subpixel-accurate operations.
    
    Attributes:
        physical_extent: The physical region being imaged
        oversampling_factor: Ratio of internal pixels to sensor pixels
        sensor_pixel_pitch_um: Physical size of sensor pixels in micrometers
        magnification: Optical magnification (object to image space)
    """
    physical_extent: PhysicalExtent
    oversampling_factor: int
    sensor_pixel_pitch_um: float
    magnification: float
    
    def __post_init__(self):
        # Compute derived values
        self._sensor_pixel_pitch_mm = self.sensor_pixel_pitch_um / 1000.0
        
        # Object-space pixel pitch (what each sensor pixel sees in the scene)
        self._object_pixel_pitch_mm = self._sensor_pixel_pitch_mm / self.magnification
        
        # Internal oversampled pixel pitch in object space
        self._internal_pixel_pitch_mm = self._object_pixel_pitch_mm / self.oversampling_factor
    
    @property
    def internal_resolution(self) -> Tuple[int, int]:
        """Resolution of the internal oversampled representation (width, height)."""
        width_px = int(np.ceil(self.physical_extent.width_mm / self._internal_pixel_pitch_mm))
        height_px = int(np.ceil(self.physical_extent.height_mm / self._internal_pixel_pitch_mm))
        return (width_px, height_px)
    
    @property
    def sensor_resolution(self) -> Tuple[int, int]:
        """Resolution at the sensor level (width, height)."""
        internal_w, internal_h = self.internal_resolution
        return (internal_w // self.oversampling_factor, internal_h // self.oversampling_factor)
    
    @property
    def internal_pixel_pitch_mm(self) -> float:
        """Size of internal pixels in object space (mm)."""
        return self._internal_pixel_pitch_mm
    
    @property
    def object_pixel_pitch_mm(self) -> float:
        """Size of sensor pixels in object space (mm)."""
        return self._object_pixel_pitch_mm
    
    def mm_to_internal_pixels(self, distance_mm: float) -> float:
        """Convert a distance in mm to internal pixel units."""
        return distance_mm / self._internal_pixel_pitch_mm
    
    def internal_pixels_to_mm(self, distance_px: float) -> float:
        """Convert internal pixel units to mm."""
        return distance_px * self._internal_pixel_pitch_mm
    
    def mm_to_sensor_pixels(self, distance_mm: float) -> float:
        """Convert a distance in mm to sensor pixel units."""
        return distance_mm / self._object_pixel_pitch_mm
    
    def physical_to_internal_coords(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """
        Convert physical coordinates (mm) to internal pixel coordinates.
        
        Origin is at top-left of the physical extent.
        """
        x_px = (x_mm - self.physical_extent.left_mm) / self._internal_pixel_pitch_mm
        y_px = (y_mm - self.physical_extent.top_mm) / self._internal_pixel_pitch_mm
        return (x_px, y_px)
    
    def internal_to_physical_coords(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """
        Convert internal pixel coordinates to physical coordinates (mm).
        """
        x_mm = x_px * self._internal_pixel_pitch_mm + self.physical_extent.left_mm
        y_mm = y_px * self._internal_pixel_pitch_mm + self.physical_extent.top_mm
        return (x_mm, y_mm)
    
    def create_coordinate_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create meshgrid of physical coordinates for the internal representation.
        
        Returns:
            (X_mm, Y_mm): 2D arrays of physical coordinates
        """
        width_px, height_px = self.internal_resolution
        
        x = np.linspace(
            self.physical_extent.left_mm + self._internal_pixel_pitch_mm / 2,
            self.physical_extent.right_mm - self._internal_pixel_pitch_mm / 2,
            width_px
        )
        y = np.linspace(
            self.physical_extent.top_mm + self._internal_pixel_pitch_mm / 2,
            self.physical_extent.bottom_mm - self._internal_pixel_pitch_mm / 2,
            height_px
        )
        
        return np.meshgrid(x, y)
