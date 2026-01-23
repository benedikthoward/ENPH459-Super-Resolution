"""
Residual warp calibration for Optotune beam shifter.

The physical Snell's law model provides a first-order approximation of the
field-dependent warp. Real systems may have additional effects (lens distortion,
plate imperfections, mounting errors) that can be captured through calibration.

This module provides tools to:
1. Fit polynomial residual warp from calibration images (dot grid)
2. Apply residual correction on top of the physical model
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.optimize import least_squares


@dataclass
class ResidualWarpCalibration:
    """
    Polynomial residual warp correction.
    
    Models residual displacement as a polynomial function of field position:
    
    dx_residual = Σ c_x[i,j] * x^i * y^j
    dy_residual = Σ c_y[i,j] * x^i * y^j
    
    Attributes:
        order: Maximum polynomial order
        coeffs_x: Coefficient array for X displacement
        coeffs_y: Coefficient array for Y displacement
        enabled: Whether to apply the correction
        calibration_points: Stored calibration data (for reference)
    """
    order: int = 3
    coeffs_x: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
    coeffs_y: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
    enabled: bool = False
    calibration_points: Optional[Dict] = None
    
    def __post_init__(self):
        if self.coeffs_x.shape[0] != self.order + 1:
            self.coeffs_x = np.zeros((self.order + 1, self.order + 1))
        if self.coeffs_y.shape[0] != self.order + 1:
            self.coeffs_y = np.zeros((self.order + 1, self.order + 1))
    
    def evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate residual warp at given positions.
        
        Args:
            x: X coordinates (normalized to [-1, 1])
            y: Y coordinates (normalized to [-1, 1])
        
        Returns:
            (dx_residual, dy_residual): Residual displacement arrays
        """
        if not self.enabled:
            return (np.zeros_like(x), np.zeros_like(y))
        
        dx = np.zeros_like(x)
        dy = np.zeros_like(y)
        
        for i in range(self.order + 1):
            for j in range(self.order + 1):
                term = (x ** i) * (y ** j)
                dx += self.coeffs_x[i, j] * term
                dy += self.coeffs_y[i, j] * term
        
        return (dx, dy)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary for YAML storage."""
        return {
            'order': self.order,
            'enabled': self.enabled,
            'coeffs_x': self.coeffs_x.tolist(),
            'coeffs_y': self.coeffs_y.tolist(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ResidualWarpCalibration':
        """Load from dictionary."""
        return cls(
            order=data.get('order', 3),
            enabled=data.get('enabled', False),
            coeffs_x=np.array(data.get('coeffs_x', np.zeros((4, 4)))),
            coeffs_y=np.array(data.get('coeffs_y', np.zeros((4, 4)))),
        )


def fit_residual_warp(
    measured_positions: np.ndarray,
    expected_positions: np.ndarray,
    model_displacements: np.ndarray,
    order: int = 3,
    image_extent: Tuple[float, float] = (1.0, 1.0),
) -> ResidualWarpCalibration:
    """
    Fit residual warp polynomial from calibration measurements.
    
    Given measured feature positions, expected positions, and model-predicted
    displacements, fits a polynomial to capture the residual error.
    
    Args:
        measured_positions: (N, 2) array of measured (x, y) positions
        expected_positions: (N, 2) array of expected (x, y) positions
        model_displacements: (N, 2) array of model-predicted (dx, dy)
        order: Polynomial order for fitting
        image_extent: (width, height) for coordinate normalization
    
    Returns:
        Fitted ResidualWarpCalibration instance
    """
    # Compute actual displacements
    actual_dx = measured_positions[:, 0] - expected_positions[:, 0]
    actual_dy = measured_positions[:, 1] - expected_positions[:, 1]
    
    # Compute residuals (actual - model)
    residual_dx = actual_dx - model_displacements[:, 0]
    residual_dy = actual_dy - model_displacements[:, 1]
    
    # Normalize coordinates to [-1, 1]
    x_norm = 2 * expected_positions[:, 0] / image_extent[0] - 1
    y_norm = 2 * expected_positions[:, 1] / image_extent[1] - 1
    
    # Build design matrix for polynomial fitting
    n_terms = (order + 1) ** 2
    design = np.zeros((len(x_norm), n_terms))
    
    idx = 0
    for i in range(order + 1):
        for j in range(order + 1):
            design[:, idx] = (x_norm ** i) * (y_norm ** j)
            idx += 1
    
    # Fit using least squares
    coeffs_x_flat, _, _, _ = np.linalg.lstsq(design, residual_dx, rcond=None)
    coeffs_y_flat, _, _, _ = np.linalg.lstsq(design, residual_dy, rcond=None)
    
    # Reshape to 2D arrays
    coeffs_x = coeffs_x_flat.reshape((order + 1, order + 1))
    coeffs_y = coeffs_y_flat.reshape((order + 1, order + 1))
    
    calibration = ResidualWarpCalibration(
        order=order,
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
        enabled=True,
        calibration_points={
            'measured': measured_positions.tolist(),
            'expected': expected_positions.tolist(),
            'n_points': len(measured_positions),
        }
    )
    
    return calibration


def detect_calibration_dots(
    image: np.ndarray,
    expected_grid: Tuple[int, int] = (10, 10),
    dot_size_range: Tuple[int, int] = (5, 50),
) -> np.ndarray:
    """
    Detect calibration dot positions in an image.
    
    Uses blob detection to find circular calibration markers.
    
    Args:
        image: Grayscale calibration image
        expected_grid: Expected (rows, cols) of dot grid
        dot_size_range: (min, max) expected dot diameter in pixels
    
    Returns:
        (N, 2) array of detected (x, y) positions, sorted by grid position
    """
    from skimage.feature import blob_dog
    from scipy.spatial.distance import cdist
    
    # Normalize image
    img_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
    
    # Detect blobs
    min_sigma = dot_size_range[0] / 4
    max_sigma = dot_size_range[1] / 4
    
    blobs = blob_dog(
        img_norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold=0.1,
    )
    
    if len(blobs) == 0:
        return np.array([]).reshape(0, 2)
    
    # Extract positions (blob_dog returns y, x, sigma)
    positions = blobs[:, :2][:, ::-1]  # Convert to (x, y)
    
    # Sort by position to match expected grid
    # Sort first by y (row), then by x (column)
    sorted_indices = np.lexsort((positions[:, 0], positions[:, 1]))
    positions = positions[sorted_indices]
    
    return positions


def generate_expected_grid(
    grid_size: Tuple[int, int],
    spacing_mm: float,
    image_size: Tuple[int, int],
    pixel_pitch_mm: float,
) -> np.ndarray:
    """
    Generate expected dot grid positions.
    
    Args:
        grid_size: (rows, cols) of the dot grid
        spacing_mm: Spacing between dots in mm
        image_size: (width, height) of the image in pixels
        pixel_pitch_mm: Size of each pixel in mm
    
    Returns:
        (N, 2) array of expected (x, y) positions in pixels
    """
    rows, cols = grid_size
    width, height = image_size
    
    # Generate grid centered in image
    spacing_px = spacing_mm / pixel_pitch_mm
    
    x_start = width / 2 - (cols - 1) * spacing_px / 2
    y_start = height / 2 - (rows - 1) * spacing_px / 2
    
    positions = []
    for r in range(rows):
        for c in range(cols):
            x = x_start + c * spacing_px
            y = y_start + r * spacing_px
            positions.append([x, y])
    
    return np.array(positions)
