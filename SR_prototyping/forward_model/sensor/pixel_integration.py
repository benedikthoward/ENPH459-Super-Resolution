"""
Pixel integration (downsampling from oversampled representation).

Simulates how sensor pixels integrate light over their sensitive area.
This converts the high-resolution internal representation to sensor resolution.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy import ndimage


@dataclass
class PixelIntegrator:
    """
    Handles pixel integration from oversampled to sensor resolution.
    
    The integration can use simple box averaging or more sophisticated
    methods that account for pixel response non-uniformity.
    
    Attributes:
        oversampling_factor: Ratio of internal to sensor resolution
        integration_mode: 'box' for uniform averaging, 'gaussian' for weighted
        fill_factor: Fraction of pixel area that is light-sensitive (0-1)
    """
    oversampling_factor: int
    integration_mode: str = 'box'
    fill_factor: float = 1.0
    
    def __post_init__(self):
        if self.fill_factor <= 0 or self.fill_factor > 1:
            raise ValueError(f"fill_factor must be in (0, 1], got {self.fill_factor}")
    
    def integrate(
        self,
        image: np.ndarray,
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Integrate oversampled image to sensor resolution.
        
        Args:
            image: Oversampled input image (2D or 3D for color)
            target_shape: Optional (H, W) output shape. If None, computed
                         from oversampling factor.
        
        Returns:
            Integrated image at sensor resolution
        """
        if target_shape is None:
            h, w = image.shape[:2]
            target_shape = (h // self.oversampling_factor, 
                           w // self.oversampling_factor)
        
        if self.integration_mode == 'box':
            return self._box_integrate(image, target_shape)
        elif self.integration_mode == 'gaussian':
            return self._gaussian_integrate(image, target_shape)
        else:
            raise ValueError(f"Unknown integration mode: {self.integration_mode}")
    
    def _box_integrate(
        self,
        image: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Simple box (uniform) integration."""
        h_out, w_out = target_shape
        osf = self.oversampling_factor
        
        # Crop to exact multiple of target shape
        h_crop = h_out * osf
        w_crop = w_out * osf
        image = image[:h_crop, :w_crop]
        
        if image.ndim == 2:
            # Reshape and average
            reshaped = image.reshape(h_out, osf, w_out, osf)
            integrated = reshaped.mean(axis=(1, 3))
        else:
            # Handle color images
            n_channels = image.shape[2]
            integrated = np.zeros((h_out, w_out, n_channels), dtype=np.float32)
            for c in range(n_channels):
                reshaped = image[:, :, c].reshape(h_out, osf, w_out, osf)
                integrated[:, :, c] = reshaped.mean(axis=(1, 3))
        
        # Apply fill factor
        integrated *= self.fill_factor
        
        return integrated
    
    def _gaussian_integrate(
        self,
        image: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Gaussian-weighted integration for smoother response."""
        h_out, w_out = target_shape
        osf = self.oversampling_factor
        
        # Create Gaussian weight kernel for pixel response
        # The kernel size is the oversampling factor
        sigma = osf / 4.0  # Adjust for desired response shape
        
        x = np.arange(osf) - osf / 2 + 0.5
        y = np.arange(osf) - osf / 2 + 0.5
        xx, yy = np.meshgrid(x, y)
        
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel *= self.fill_factor
        kernel /= kernel.sum() * (osf * osf)  # Normalize
        
        # Crop image
        h_crop = h_out * osf
        w_crop = w_out * osf
        image = image[:h_crop, :w_crop]
        
        if image.ndim == 2:
            integrated = np.zeros((h_out, w_out), dtype=np.float32)
            for i in range(h_out):
                for j in range(w_out):
                    patch = image[i*osf:(i+1)*osf, j*osf:(j+1)*osf]
                    integrated[i, j] = np.sum(patch * kernel)
        else:
            n_channels = image.shape[2]
            integrated = np.zeros((h_out, w_out, n_channels), dtype=np.float32)
            for c in range(n_channels):
                for i in range(h_out):
                    for j in range(w_out):
                        patch = image[i*osf:(i+1)*osf, j*osf:(j+1)*osf, c]
                        integrated[i, j, c] = np.sum(patch * kernel)
        
        return integrated


def integrate_pixels(
    image: np.ndarray,
    oversampling_factor: int,
    mode: str = 'box',
    fill_factor: float = 1.0,
    target_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Convenience function for pixel integration.
    
    Args:
        image: Oversampled input image
        oversampling_factor: Ratio of input to output resolution
        mode: Integration mode ('box' or 'gaussian')
        fill_factor: Pixel fill factor
        target_shape: Optional output shape
    
    Returns:
        Integrated image at sensor resolution
    """
    integrator = PixelIntegrator(
        oversampling_factor=oversampling_factor,
        integration_mode=mode,
        fill_factor=fill_factor,
    )
    return integrator.integrate(image, target_shape)


def downsample_antialiased(
    image: np.ndarray,
    factor: int,
    filter_sigma: Optional[float] = None,
) -> np.ndarray:
    """
    Downsample with antialiasing (low-pass filtering before subsampling).
    
    This is an alternative to box integration that uses explicit filtering.
    
    Args:
        image: Input image
        factor: Downsampling factor
        filter_sigma: Gaussian filter sigma. If None, set to factor/2.
    
    Returns:
        Downsampled image
    """
    if filter_sigma is None:
        filter_sigma = factor / 2.0
    
    # Apply Gaussian low-pass filter
    if image.ndim == 2:
        filtered = ndimage.gaussian_filter(image, sigma=filter_sigma)
    else:
        filtered = np.zeros_like(image)
        for c in range(image.shape[2]):
            filtered[:, :, c] = ndimage.gaussian_filter(
                image[:, :, c], sigma=filter_sigma
            )
    
    # Subsample
    return filtered[::factor, ::factor]
