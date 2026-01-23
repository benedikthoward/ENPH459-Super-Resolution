"""
Thin lens and geometric optics modeling.

Provides calculations for magnification, image distance, and chief ray angles
for field-dependent computations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


class LensModel(ABC):
    """Abstract base class for lens models."""
    
    @abstractmethod
    def magnification(self) -> float:
        """Return the optical magnification (image size / object size)."""
        pass
    
    @abstractmethod
    def image_distance_mm(self) -> float:
        """Return the image distance in mm."""
        pass
    
    @abstractmethod
    def chief_ray_angle(self, field_position_mm: float) -> float:
        """
        Compute the chief ray angle at a given field position.
        
        Args:
            field_position_mm: Radial distance from optical axis in image space (mm)
        
        Returns:
            Chief ray angle in radians
        """
        pass


@dataclass
class ThinLensModel(LensModel):
    """
    Thin lens approximation for geometric optics calculations.
    
    Uses the thin lens equation: 1/f = 1/d_o + 1/d_i
    Magnification: m = -d_i / d_o
    
    Attributes:
        focal_length_mm: Lens focal length in mm
        object_distance_mm: Distance from object to lens in mm
        f_number: F-number (f/D) of the lens
        wavelength_nm: Primary wavelength for diffraction calculations
    """
    focal_length_mm: float
    object_distance_mm: float
    f_number: float = 2.8
    wavelength_nm: float = 550.0
    
    def __post_init__(self):
        # Validate inputs
        if self.object_distance_mm <= self.focal_length_mm:
            raise ValueError(
                f"Object distance ({self.object_distance_mm}mm) must be greater than "
                f"focal length ({self.focal_length_mm}mm) for real image formation."
            )
        
        # Compute image distance from thin lens equation
        self._image_distance_mm = 1.0 / (
            1.0 / self.focal_length_mm - 1.0 / self.object_distance_mm
        )
        
        # Compute magnification (negative for inverted image, but we use absolute)
        self._magnification = abs(self._image_distance_mm / self.object_distance_mm)
        
        # Compute aperture diameter
        self._aperture_mm = self.focal_length_mm / self.f_number
    
    def magnification(self) -> float:
        """Return the optical magnification (absolute value)."""
        return self._magnification
    
    def image_distance_mm(self) -> float:
        """Return the image distance in mm."""
        return self._image_distance_mm
    
    @property
    def aperture_diameter_mm(self) -> float:
        """Return the aperture diameter in mm."""
        return self._aperture_mm
    
    @property
    def wavelength_mm(self) -> float:
        """Return the wavelength in mm."""
        return self.wavelength_nm / 1e6
    
    def chief_ray_angle(self, field_position_mm: float) -> float:
        """
        Compute the chief ray angle at a given field position in image space.
        
        For a thin lens, the chief ray passes through the center of the lens,
        so the angle is simply arctan(field_position / image_distance).
        
        Args:
            field_position_mm: Radial distance from optical axis in image space (mm)
        
        Returns:
            Chief ray angle in radians
        """
        return np.arctan(field_position_mm / self._image_distance_mm)
    
    def chief_ray_angle_field(
        self, 
        x_mm: np.ndarray, 
        y_mm: np.ndarray
    ) -> np.ndarray:
        """
        Compute chief ray angles for a field of positions.
        
        Args:
            x_mm: X coordinates in image space (mm)
            y_mm: Y coordinates in image space (mm)
        
        Returns:
            2D array of chief ray angles in radians
        """
        r_mm = np.sqrt(x_mm**2 + y_mm**2)
        return np.arctan(r_mm / self._image_distance_mm)
    
    def diffraction_limited_spot_size_um(self) -> float:
        """
        Compute the diffraction-limited Airy disk diameter (first zero).
        
        The Airy disk diameter is: d = 2.44 * wavelength * f_number
        """
        return 2.44 * (self.wavelength_nm / 1000.0) * self.f_number
    
    def na_image_side(self) -> float:
        """Numerical aperture on the image side."""
        return 1.0 / (2.0 * self.f_number)
    
    def na_object_side(self) -> float:
        """Numerical aperture on the object side."""
        return self.na_image_side() / self._magnification
    
    def depth_of_field_mm(self, circle_of_confusion_um: Optional[float] = None) -> float:
        """
        Calculate the depth of field in object space.
        
        Args:
            circle_of_confusion_um: Acceptable blur diameter in micrometers.
                                   Defaults to diffraction limit.
        
        Returns:
            Total depth of field in mm
        """
        if circle_of_confusion_um is None:
            circle_of_confusion_um = self.diffraction_limited_spot_size_um()
        
        coc_mm = circle_of_confusion_um / 1000.0
        
        # Hyperfocal distance
        H = (self.focal_length_mm ** 2) / (self.f_number * coc_mm)
        
        # Near and far focus distances
        d_o = self.object_distance_mm
        d_near = (H * d_o) / (H + (d_o - self.focal_length_mm))
        d_far = (H * d_o) / (H - (d_o - self.focal_length_mm))
        
        if d_far < 0:  # Beyond hyperfocal distance
            return float('inf')
        
        return d_far - d_near
    
    def object_to_image_coords(
        self,
        x_obj_mm: np.ndarray,
        y_obj_mm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map object-space coordinates to image-space coordinates.
        
        Args:
            x_obj_mm: X coordinates in object space (mm)
            y_obj_mm: Y coordinates in object space (mm)
        
        Returns:
            (x_img_mm, y_img_mm): Coordinates in image space
        """
        # Simple scaling by magnification (ignoring image inversion)
        x_img = x_obj_mm * self._magnification
        y_img = y_obj_mm * self._magnification
        return (x_img, y_img)
