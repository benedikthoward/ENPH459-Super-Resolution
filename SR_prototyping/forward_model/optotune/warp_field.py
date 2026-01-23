"""
Image warp field generation and application.

Creates displacement maps from the Optotune plate model and applies them
to images using high-quality interpolation.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy import ndimage

from .plate_model import PlateModel
from ..scene.scene_units import SceneUnits


@dataclass
class WarpField:
    """
    Displacement field for image warping.
    
    Stores the (dx, dy) displacement at each pixel and provides methods
    to apply the warp using various interpolation schemes.
    
    Attributes:
        dx: X displacement in pixels (same shape as image)
        dy: Y displacement in pixels (same shape as image)
        dx_mm: X displacement in millimeters
        dy_mm: Y displacement in millimeters
    """
    dx: np.ndarray
    dy: np.ndarray
    dx_mm: np.ndarray
    dy_mm: np.ndarray
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.dx.shape
    
    @property
    def magnitude(self) -> np.ndarray:
        """Displacement magnitude in pixels."""
        return np.sqrt(self.dx**2 + self.dy**2)
    
    @property
    def magnitude_mm(self) -> np.ndarray:
        """Displacement magnitude in millimeters."""
        return np.sqrt(self.dx_mm**2 + self.dy_mm**2)
    
    @property
    def max_shift_px(self) -> float:
        """Maximum displacement in pixels."""
        return float(self.magnitude.max())
    
    @property
    def mean_shift_px(self) -> float:
        """Mean displacement in pixels."""
        return float(self.magnitude.mean())
    
    def apply(
        self,
        image: np.ndarray,
        order: int = 3,
        mode: str = 'constant',
        cval: float = 0.0,
    ) -> np.ndarray:
        """
        Apply the warp field to an image.
        
        Uses scipy's map_coordinates for high-quality interpolation.
        
        Args:
            image: Input image (2D or 3D for color)
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)
            mode: How to handle boundaries ('constant', 'nearest', 'reflect')
            cval: Value for constant mode
        
        Returns:
            Warped image
        """
        # Create sampling coordinates
        # The warp is additive: new_pos = old_pos + displacement
        y_coords, x_coords = np.mgrid[:image.shape[0], :image.shape[1]]
        
        # Add displacement to get source coordinates
        src_y = y_coords.astype(np.float64) - self.dy
        src_x = x_coords.astype(np.float64) - self.dx
        
        if image.ndim == 2:
            return ndimage.map_coordinates(
                image,
                [src_y, src_x],
                order=order,
                mode=mode,
                cval=cval,
            )
        else:
            # Handle color images
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = ndimage.map_coordinates(
                    image[:, :, c],
                    [src_y, src_x],
                    order=order,
                    mode=mode,
                    cval=cval,
                )
            return result
    
    def apply_inverse(
        self,
        image: np.ndarray,
        order: int = 3,
        mode: str = 'constant',
        cval: float = 0.0,
    ) -> np.ndarray:
        """
        Apply the inverse warp (for reconstruction).
        
        Args:
            image: Input image
            order: Interpolation order
            mode: Boundary mode
            cval: Constant value
        
        Returns:
            Inverse-warped image
        """
        y_coords, x_coords = np.mgrid[:image.shape[0], :image.shape[1]]
        
        # Inverse warp: subtract displacement
        src_y = y_coords.astype(np.float64) + self.dy
        src_x = x_coords.astype(np.float64) + self.dx
        
        if image.ndim == 2:
            return ndimage.map_coordinates(
                image, [src_y, src_x], order=order, mode=mode, cval=cval
            )
        else:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = ndimage.map_coordinates(
                    image[:, :, c], [src_y, src_x], order=order, mode=mode, cval=cval
                )
            return result
    
    def to_flow_visualization(self, scale: float = 1.0) -> np.ndarray:
        """
        Convert to HSV flow visualization for debugging.
        
        Hue encodes direction, saturation/value encode magnitude.
        
        Args:
            scale: Scale factor for visualization
        
        Returns:
            RGB image (H, W, 3) as uint8
        """
        import colorsys
        
        # Compute angle and magnitude
        angle = np.arctan2(self.dy, self.dx)
        mag = self.magnitude * scale
        
        # Normalize magnitude to [0, 1]
        max_mag = mag.max() if mag.max() > 0 else 1.0
        mag_norm = mag / max_mag
        
        # Create HSV image
        h = (angle + np.pi) / (2 * np.pi)  # Map [-π, π] to [0, 1]
        s = np.ones_like(h)
        v = mag_norm
        
        # Convert to RGB
        rgb = np.zeros((*self.dx.shape, 3), dtype=np.uint8)
        for i in range(self.dx.shape[0]):
            for j in range(self.dx.shape[1]):
                r, g, b = colorsys.hsv_to_rgb(h[i, j], s[i, j], v[i, j])
                rgb[i, j] = [int(r * 255), int(g * 255), int(b * 255)]
        
        return rgb


def create_warp_field(
    plate: PlateModel,
    scene_units: SceneUnits,
    image_distance_mm: float,
    target_shape: Optional[Tuple[int, int]] = None,
) -> WarpField:
    """
    Create a warp field from an Optotune plate model.
    
    Args:
        plate: Optotune plate model with current tilt settings
        scene_units: Scene coordinate system
        image_distance_mm: Distance from lens to image/sensor plane
        target_shape: Output shape (H, W). If None, uses internal resolution.
    
    Returns:
        WarpField instance with displacement maps
    """
    if target_shape is None:
        target_shape = scene_units.internal_resolution[::-1]  # (H, W)
    
    # Create coordinate grids in image space (mm)
    # Need to map internal coordinates to image plane coordinates
    height, width = target_shape
    
    # Physical coordinates in object space
    x_obj = np.linspace(
        scene_units.physical_extent.left_mm,
        scene_units.physical_extent.right_mm,
        width,
    )
    y_obj = np.linspace(
        scene_units.physical_extent.top_mm,
        scene_units.physical_extent.bottom_mm,
        height,
    )
    x_obj_grid, y_obj_grid = np.meshgrid(x_obj, y_obj)
    
    # Convert to image space coordinates (apply magnification)
    x_img_mm = x_obj_grid * scene_units.magnification
    y_img_mm = y_obj_grid * scene_units.magnification
    
    # Compute displacement field in mm
    dx_mm, dy_mm = plate.shift_field(x_img_mm, y_img_mm, image_distance_mm)
    
    # Convert to internal pixel units
    # Displacement in image space needs to be converted to object space pixels
    pixel_pitch_mm = scene_units.internal_pixel_pitch_mm
    dx_px = dx_mm / scene_units.magnification / pixel_pitch_mm
    dy_px = dy_mm / scene_units.magnification / pixel_pitch_mm
    
    return WarpField(
        dx=dx_px.astype(np.float32),
        dy=dy_px.astype(np.float32),
        dx_mm=dx_mm.astype(np.float32),
        dy_mm=dy_mm.astype(np.float32),
    )


def create_uniform_warp(
    dx_px: float,
    dy_px: float,
    shape: Tuple[int, int],
) -> WarpField:
    """
    Create a uniform (constant) warp field.
    
    Useful for first-order approximations or testing.
    
    Args:
        dx_px: X displacement in pixels
        dy_px: Y displacement in pixels
        shape: Output shape (H, W)
    
    Returns:
        WarpField with constant displacement
    """
    dx = np.full(shape, dx_px, dtype=np.float32)
    dy = np.full(shape, dy_px, dtype=np.float32)
    
    # For uniform warp, mm values are not meaningful without context
    return WarpField(dx=dx, dy=dy, dx_mm=dx, dy_mm=dy)
