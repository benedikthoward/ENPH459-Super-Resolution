"""
Motion blur modeling for moving samples.

Creates directional blur kernels based on sample velocity, exposure time,
and optical magnification.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve


@dataclass
class MotionBlurModel:
    """
    Linear motion blur model.
    
    Models the blur caused by constant-velocity motion during exposure.
    The blur is a line kernel oriented in the direction of motion.
    
    Attributes:
        velocity_mm_s: Sample velocity vector (vx, vy) in mm/s (object space)
        exposure_time_s: Exposure duration in seconds
        magnification: Optical magnification
    """
    velocity_mm_s: Tuple[float, float]
    exposure_time_s: float
    magnification: float
    
    @property
    def motion_distance_object_mm(self) -> float:
        """Total motion distance in object space (mm)."""
        vx, vy = self.velocity_mm_s
        return np.sqrt(vx**2 + vy**2) * self.exposure_time_s
    
    @property
    def motion_distance_image_mm(self) -> float:
        """Total motion distance in image space (mm)."""
        return self.motion_distance_object_mm * self.magnification
    
    @property
    def motion_angle_rad(self) -> float:
        """Direction of motion in radians (0 = positive x)."""
        vx, vy = self.velocity_mm_s
        return np.arctan2(vy, vx)
    
    def motion_distance_pixels(self, pixel_pitch_um: float) -> float:
        """
        Motion distance in sensor pixels.
        
        Args:
            pixel_pitch_um: Sensor pixel pitch in micrometers
        
        Returns:
            Motion distance in pixels
        """
        return (self.motion_distance_image_mm * 1000) / pixel_pitch_um
    
    def generate_kernel(
        self,
        pixel_pitch_um: float,
        in_object_space: bool = True,
    ) -> np.ndarray:
        """
        Generate the motion blur kernel.
        
        Args:
            pixel_pitch_um: Pixel pitch for the kernel (micrometers)
            in_object_space: If True, pixel_pitch is in object space
                           (internal oversampled representation).
                           If False, it's in image space (sensor).
        
        Returns:
            2D motion blur kernel normalized to sum to 1
        """
        if in_object_space:
            # Convert object-space pixel pitch to image space
            effective_pitch_um = pixel_pitch_um * self.magnification
            motion_mm = self.motion_distance_object_mm
        else:
            effective_pitch_um = pixel_pitch_um
            motion_mm = self.motion_distance_image_mm
        
        # Motion in pixels
        motion_px = (motion_mm * 1000) / effective_pitch_um
        
        if motion_px < 0.5:
            # Motion is subpixel - return delta function
            return np.array([[1.0]], dtype=np.float32)
        
        # Create line kernel
        kernel_size = int(np.ceil(motion_px)) * 2 + 1
        center = kernel_size // 2
        
        # Create antialiased line
        angle = self.motion_angle_rad
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Sample points along the line
        n_samples = max(int(motion_px * 4), 10)  # Oversample for antialiasing
        t = np.linspace(-motion_px/2, motion_px/2, n_samples)
        
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        
        for ti in t:
            x = center + ti * dx
            y = center + ti * dy
            
            # Bilinear interpolation for antialiased kernel
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            x1, y1 = x0 + 1, y0 + 1
            
            if 0 <= x0 < kernel_size and 0 <= y0 < kernel_size:
                wx = x - x0
                wy = y - y0
                
                kernel[y0, x0] += (1 - wx) * (1 - wy)
                if x1 < kernel_size:
                    kernel[y0, x1] += wx * (1 - wy)
                if y1 < kernel_size:
                    kernel[y1, x0] += (1 - wx) * wy
                if x1 < kernel_size and y1 < kernel_size:
                    kernel[y1, x1] += wx * wy
        
        # Normalize
        kernel = kernel / kernel.sum()
        
        # Crop to tight bounds
        kernel = _crop_kernel(kernel)
        
        return kernel
    
    def apply(
        self,
        image: np.ndarray,
        pixel_pitch_um: float,
        in_object_space: bool = True,
    ) -> np.ndarray:
        """
        Apply motion blur to an image.
        
        Args:
            image: Input image (2D or 3D for color)
            pixel_pitch_um: Pixel pitch of the image
            in_object_space: Whether pixel pitch is in object space
        
        Returns:
            Motion-blurred image
        """
        kernel = self.generate_kernel(pixel_pitch_um, in_object_space)
        
        if image.ndim == 2:
            return fftconvolve(image, kernel, mode='same')
        else:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = fftconvolve(image[:, :, c], kernel, mode='same')
            return result


def _crop_kernel(kernel: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Crop kernel to remove near-zero borders."""
    # Find non-zero region
    rows = np.any(kernel > threshold, axis=1)
    cols = np.any(kernel > threshold, axis=0)
    
    if not rows.any() or not cols.any():
        return np.array([[1.0]], dtype=np.float32)
    
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    
    # Ensure odd size for symmetry
    height = r_max - r_min + 1
    width = c_max - c_min + 1
    
    if height % 2 == 0:
        r_max = min(r_max + 1, kernel.shape[0] - 1)
    if width % 2 == 0:
        c_max = min(c_max + 1, kernel.shape[1] - 1)
    
    return kernel[r_min:r_max+1, c_min:c_max+1]


def apply_motion_blur(
    image: np.ndarray,
    velocity_mm_s: Tuple[float, float],
    exposure_time_s: float,
    magnification: float,
    pixel_pitch_um: float,
    in_object_space: bool = True,
) -> np.ndarray:
    """
    Convenience function to apply motion blur.
    
    Args:
        image: Input image
        velocity_mm_s: Sample velocity (vx, vy) in mm/s
        exposure_time_s: Exposure time in seconds
        magnification: Optical magnification
        pixel_pitch_um: Pixel pitch of the image
        in_object_space: Whether pixel pitch is in object space
    
    Returns:
        Motion-blurred image
    """
    model = MotionBlurModel(
        velocity_mm_s=velocity_mm_s,
        exposure_time_s=exposure_time_s,
        magnification=magnification,
    )
    return model.apply(image, pixel_pitch_um, in_object_space)


def create_motion_kernel_direct(
    length_px: float,
    angle_rad: float,
) -> np.ndarray:
    """
    Create a motion blur kernel directly from length and angle.
    
    Args:
        length_px: Blur length in pixels
        angle_rad: Blur direction in radians
    
    Returns:
        Motion blur kernel
    """
    if length_px < 0.5:
        return np.array([[1.0]], dtype=np.float32)
    
    kernel_size = int(np.ceil(length_px)) * 2 + 1
    center = kernel_size // 2
    
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    
    n_samples = max(int(length_px * 4), 10)
    t = np.linspace(-length_px/2, length_px/2, n_samples)
    
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    
    for ti in t:
        x = center + ti * dx
        y = center + ti * dy
        
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = x0 + 1, y0 + 1
        
        if 0 <= x0 < kernel_size and 0 <= y0 < kernel_size:
            wx = x - x0
            wy = y - y0
            
            kernel[y0, x0] += (1 - wx) * (1 - wy)
            if x1 < kernel_size:
                kernel[y0, x1] += wx * (1 - wy)
            if y1 < kernel_size:
                kernel[y1, x0] += (1 - wx) * wy
            if x1 < kernel_size and y1 < kernel_size:
                kernel[y1, x1] += wx * wy
    
    kernel = kernel / kernel.sum()
    return _crop_kernel(kernel)
