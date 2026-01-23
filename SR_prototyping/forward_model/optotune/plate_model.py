"""
Tilted parallel plate beam shifter physics.

Implements Snell's law ray tracing through a tilted glass plate to compute
the lateral displacement as a function of plate tilt and ray incidence angle.

Key equation for lateral displacement through tilted plate:
    Δ = t * sin(θ) * (1 - sqrt((1 - sin²(θ)) / (n² - sin²(θ))))

Where:
    t = plate thickness
    n = refractive index
    θ = incidence angle (plate tilt + chief ray angle)
"""

from dataclasses import dataclass
from typing import Tuple, Union
import numpy as np


@dataclass
class PlateModel:
    """
    Model of a tilted parallel plate beam shifter.
    
    The plate causes lateral displacement of transmitted rays according to
    Snell's law refraction at both surfaces. The displacement depends on:
    - Plate thickness and refractive index
    - Plate tilt angles (X and Y axes)
    - Chief ray angle (varies across field)
    
    Attributes:
        thickness_mm: Plate thickness in millimeters
        refractive_index: Glass refractive index (typically 1.5-1.9)
        tilt_x_deg: Rotation about X axis in degrees
        tilt_y_deg: Rotation about Y axis in degrees
        placement: "converging" (after lens) or "collimated" (before lens)
    """
    thickness_mm: float
    refractive_index: float
    tilt_x_deg: float = 0.0
    tilt_y_deg: float = 0.0
    placement: str = "converging"
    
    @property
    def tilt_x_rad(self) -> float:
        return np.deg2rad(self.tilt_x_deg)
    
    @property
    def tilt_y_rad(self) -> float:
        return np.deg2rad(self.tilt_y_deg)
    
    def lateral_shift(
        self,
        incidence_angle_rad: float,
    ) -> float:
        """
        Compute lateral displacement for a single incidence angle.
        
        Uses the exact formula derived from Snell's law geometry.
        
        Args:
            incidence_angle_rad: Total incidence angle in radians
                                (plate tilt + chief ray angle)
        
        Returns:
            Lateral shift in millimeters
        """
        return compute_plate_shift(
            incidence_angle_rad,
            self.thickness_mm,
            self.refractive_index,
        )
    
    def shift_at_field_position(
        self,
        field_x_mm: float,
        field_y_mm: float,
        image_distance_mm: float,
    ) -> Tuple[float, float]:
        """
        Compute the (dx, dy) shift at a specific field position.
        
        The chief ray angle varies across the field, creating field-dependent
        displacement. This is the key effect that must be modeled for accurate
        super-resolution reconstruction.
        
        Args:
            field_x_mm: X position in image plane (mm from optical axis)
            field_y_mm: Y position in image plane (mm from optical axis)
            image_distance_mm: Distance from lens to image plane
        
        Returns:
            (dx_mm, dy_mm): Displacement components in millimeters
        """
        # Chief ray angles at this field position
        # For converging beam placement, chief ray angle = atan(field_pos / image_dist)
        if self.placement == "converging":
            chief_x = np.arctan(field_x_mm / image_distance_mm)
            chief_y = np.arctan(field_y_mm / image_distance_mm)
        else:
            # Collimated beam - chief ray is parallel to optical axis
            chief_x = 0.0
            chief_y = 0.0
        
        # Total incidence angles (plate tilt + chief ray)
        total_angle_x = self.tilt_x_rad + chief_x
        total_angle_y = self.tilt_y_rad + chief_y
        
        # Compute shifts for each tilt direction
        # The shift is perpendicular to the tilt axis
        shift_from_x_tilt = self.lateral_shift(total_angle_x)
        shift_from_y_tilt = self.lateral_shift(total_angle_y)
        
        # X tilt causes Y displacement, Y tilt causes X displacement
        dx_mm = shift_from_y_tilt * np.sign(self.tilt_y_rad + chief_y)
        dy_mm = shift_from_x_tilt * np.sign(self.tilt_x_rad + chief_x)
        
        return (dx_mm, dy_mm)
    
    def shift_field(
        self,
        x_mm: np.ndarray,
        y_mm: np.ndarray,
        image_distance_mm: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute displacement field for an array of positions.
        
        Args:
            x_mm: 2D array of X positions (mm)
            y_mm: 2D array of Y positions (mm)
            image_distance_mm: Distance from lens to image plane
        
        Returns:
            (dx_mm, dy_mm): 2D arrays of displacement components
        """
        if self.placement == "converging":
            chief_x = np.arctan(x_mm / image_distance_mm)
            chief_y = np.arctan(y_mm / image_distance_mm)
        else:
            chief_x = np.zeros_like(x_mm)
            chief_y = np.zeros_like(y_mm)
        
        # Total incidence angles
        total_x = self.tilt_x_rad + chief_x
        total_y = self.tilt_y_rad + chief_y
        
        # Compute shifts using vectorized formula
        shift_x = compute_plate_shift_vectorized(
            total_y, self.thickness_mm, self.refractive_index
        )
        shift_y = compute_plate_shift_vectorized(
            total_x, self.thickness_mm, self.refractive_index
        )
        
        # Apply signs based on angle direction
        dx_mm = shift_x * np.sign(total_y)
        dy_mm = shift_y * np.sign(total_x)
        
        return (dx_mm, dy_mm)
    
    def uniform_shift(self) -> Tuple[float, float]:
        """
        Compute the uniform (on-axis) shift with no field dependence.
        
        This is the shift at the optical axis where the chief ray angle is zero.
        Useful for first-order approximations or collimated beam placement.
        
        Returns:
            (dx_mm, dy_mm): On-axis displacement
        """
        shift_x = self.lateral_shift(self.tilt_y_rad)
        shift_y = self.lateral_shift(self.tilt_x_rad)
        
        dx_mm = shift_x * np.sign(self.tilt_y_rad) if self.tilt_y_deg != 0 else 0.0
        dy_mm = shift_y * np.sign(self.tilt_x_rad) if self.tilt_x_deg != 0 else 0.0
        
        return (dx_mm, dy_mm)


def compute_plate_shift(
    incidence_angle_rad: float,
    thickness_mm: float,
    refractive_index: float,
) -> float:
    """
    Compute lateral shift through a tilted parallel plate.
    
    Derived from Snell's law geometry:
    Δ = t * sin(θ) * (1 - cos(θ) / sqrt(n² - sin²(θ)))
    
    Equivalent form:
    Δ = t * sin(θ) * (1 - sqrt((1 - sin²(θ)) / (n² - sin²(θ))))
    
    Args:
        incidence_angle_rad: Angle of incidence in radians
        thickness_mm: Plate thickness in mm
        refractive_index: Glass refractive index
    
    Returns:
        Lateral shift in mm (always positive, direction handled separately)
    """
    theta = abs(incidence_angle_rad)
    
    if theta < 1e-10:
        return 0.0
    
    sin_theta = np.sin(theta)
    n = refractive_index
    
    # Check for total internal reflection (shouldn't happen for reasonable angles)
    if sin_theta >= n:
        raise ValueError(
            f"Incidence angle {np.rad2deg(theta):.1f}° exceeds critical angle "
            f"for n={n}"
        )
    
    # Snell's law formula for lateral displacement
    cos_theta = np.cos(theta)
    sqrt_term = np.sqrt(n**2 - sin_theta**2)
    
    shift = thickness_mm * sin_theta * (1 - cos_theta / sqrt_term)
    
    return shift


def compute_plate_shift_vectorized(
    incidence_angles_rad: np.ndarray,
    thickness_mm: float,
    refractive_index: float,
) -> np.ndarray:
    """
    Vectorized version of plate shift computation.
    
    Args:
        incidence_angles_rad: Array of incidence angles
        thickness_mm: Plate thickness
        refractive_index: Glass refractive index
    
    Returns:
        Array of lateral shifts (same shape as input)
    """
    theta = np.abs(incidence_angles_rad)
    n = refractive_index
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Avoid division by zero for small angles
    with np.errstate(divide='ignore', invalid='ignore'):
        sqrt_term = np.sqrt(n**2 - sin_theta**2)
        shift = thickness_mm * sin_theta * (1 - cos_theta / sqrt_term)
    
    # Handle zero angle case
    shift = np.where(theta < 1e-10, 0.0, shift)
    
    return shift


def shift_vs_angle_curve(
    plate: PlateModel,
    angle_range_deg: Tuple[float, float] = (-5, 5),
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate shift vs. angle curve for validation.
    
    Args:
        plate: PlateModel instance
        angle_range_deg: (min, max) angle range in degrees
        n_points: Number of sample points
    
    Returns:
        (angles_deg, shifts_mm): Arrays for plotting
    """
    angles_deg = np.linspace(angle_range_deg[0], angle_range_deg[1], n_points)
    angles_rad = np.deg2rad(angles_deg)
    
    shifts_mm = compute_plate_shift_vectorized(
        angles_rad,
        plate.thickness_mm,
        plate.refractive_index,
    ) * np.sign(angles_rad)
    
    return (angles_deg, shifts_mm)
