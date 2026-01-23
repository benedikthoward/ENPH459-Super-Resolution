"""
Sensor noise modeling.

Implements the physical noise sources in CMOS/CCD sensors:
- Photon shot noise (Poisson statistics)
- Read noise (Gaussian, from readout electronics)
- Dark current (temperature-dependent electron generation)
- Fixed pattern noise (pixel-to-pixel variations)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class SensorNoise:
    """
    Sensor noise parameters.
    
    All noise values are in electrons unless otherwise specified.
    
    Attributes:
        read_noise_e: RMS read noise in electrons
        dark_current_e_per_s: Dark current in electrons per second
        fixed_pattern_noise_percent: FPN as percentage of signal (0-100)
        quantum_efficiency: Fraction of photons converted to electrons (0-1)
    """
    read_noise_e: float = 2.5
    dark_current_e_per_s: float = 0.5
    fixed_pattern_noise_percent: float = 0.0
    quantum_efficiency: float = 0.7
    
    def __post_init__(self):
        if not 0 < self.quantum_efficiency <= 1:
            raise ValueError(f"QE must be in (0, 1], got {self.quantum_efficiency}")


@dataclass
class NoiseModel:
    """
    Complete noise model for sensor simulation.
    
    Converts photon flux to electron counts and adds noise according to
    physical noise sources.
    
    Attributes:
        noise_params: SensorNoise instance with noise parameters
        exposure_time_s: Exposure duration in seconds
        full_well_e: Maximum electron capacity per pixel
        seed: Random seed for reproducibility (None for non-deterministic)
    """
    noise_params: SensorNoise
    exposure_time_s: float
    full_well_e: float = 10000
    seed: Optional[int] = None
    
    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._fpn_map: Optional[np.ndarray] = None
    
    def apply(
        self,
        photons_per_pixel: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the complete noise model to a photon flux image.
        
        Args:
            photons_per_pixel: Input in photon counts per pixel
        
        Returns:
            Electron counts per pixel (float, not clipped)
        """
        qe = self.noise_params.quantum_efficiency
        
        # Convert photons to expected electrons
        expected_electrons = photons_per_pixel * qe
        
        # Add shot noise (Poisson distribution)
        # For large values, use Gaussian approximation for speed
        electrons = self._apply_shot_noise(expected_electrons)
        
        # Add dark current
        dark_electrons = (self.noise_params.dark_current_e_per_s * 
                         self.exposure_time_s)
        electrons += dark_electrons
        
        # Add more shot noise from dark current
        if dark_electrons > 0:
            electrons += self._rng.poisson(dark_electrons, size=electrons.shape)
        
        # Add read noise
        electrons += self._rng.normal(
            0, 
            self.noise_params.read_noise_e,
            size=electrons.shape
        )
        
        # Add fixed pattern noise
        if self.noise_params.fixed_pattern_noise_percent > 0:
            electrons = self._apply_fpn(electrons)
        
        return electrons
    
    def _apply_shot_noise(self, expected: np.ndarray) -> np.ndarray:
        """Apply shot noise using Poisson or Gaussian approximation."""
        result = np.zeros_like(expected, dtype=np.float64)
        
        # Use Poisson for small values (more accurate)
        small_mask = expected < 100
        if np.any(small_mask):
            small_vals = np.maximum(expected[small_mask], 0)
            result[small_mask] = self._rng.poisson(small_vals)
        
        # Use Gaussian for large values (faster)
        large_mask = ~small_mask
        if np.any(large_mask):
            large_vals = expected[large_mask]
            # Poisson variance equals mean
            result[large_mask] = self._rng.normal(
                large_vals,
                np.sqrt(np.maximum(large_vals, 0))
            )
        
        return result
    
    def _apply_fpn(self, electrons: np.ndarray) -> np.ndarray:
        """Apply fixed pattern noise."""
        # Generate FPN map if not already created
        if self._fpn_map is None or self._fpn_map.shape != electrons.shape:
            fpn_sigma = self.noise_params.fixed_pattern_noise_percent / 100.0
            self._fpn_map = 1.0 + self._rng.normal(0, fpn_sigma, size=electrons.shape)
        
        return electrons * self._fpn_map
    
    def apply_to_intensity(
        self,
        intensity: np.ndarray,
        photons_at_saturation: float = 10000,
    ) -> np.ndarray:
        """
        Apply noise model to normalized intensity image.
        
        Convenience method that scales intensity [0, 1] to photon counts.
        
        Args:
            intensity: Normalized intensity image (0-1 range)
            photons_at_saturation: Photon count corresponding to intensity=1
        
        Returns:
            Electron counts
        """
        photons = intensity * photons_at_saturation
        return self.apply(photons)
    
    def snr(self, signal_electrons: float) -> float:
        """
        Calculate signal-to-noise ratio at a given signal level.
        
        Args:
            signal_electrons: Signal level in electrons
        
        Returns:
            SNR value
        """
        shot_noise_sq = signal_electrons  # Poisson variance
        read_noise_sq = self.noise_params.read_noise_e ** 2
        dark_noise_sq = self.noise_params.dark_current_e_per_s * self.exposure_time_s
        
        total_noise = np.sqrt(shot_noise_sq + read_noise_sq + dark_noise_sq)
        
        if total_noise == 0:
            return float('inf')
        
        return signal_electrons / total_noise


def apply_sensor_noise(
    photons: np.ndarray,
    read_noise_e: float = 2.5,
    dark_current_e_per_s: float = 0.5,
    exposure_time_s: float = 0.001,
    quantum_efficiency: float = 0.7,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function to apply sensor noise.
    
    Args:
        photons: Photon counts per pixel
        read_noise_e: Read noise in electrons
        dark_current_e_per_s: Dark current rate
        exposure_time_s: Exposure time
        quantum_efficiency: Quantum efficiency
        seed: Random seed
    
    Returns:
        Electron counts with noise
    """
    noise_params = SensorNoise(
        read_noise_e=read_noise_e,
        dark_current_e_per_s=dark_current_e_per_s,
        quantum_efficiency=quantum_efficiency,
    )
    
    model = NoiseModel(
        noise_params=noise_params,
        exposure_time_s=exposure_time_s,
        seed=seed,
    )
    
    return model.apply(photons)


def estimate_noise_floor(
    noise_params: SensorNoise,
    exposure_time_s: float,
) -> float:
    """
    Estimate the noise floor (minimum detectable signal).
    
    Args:
        noise_params: Sensor noise parameters
        exposure_time_s: Exposure time
    
    Returns:
        Noise floor in electrons
    """
    read_noise_sq = noise_params.read_noise_e ** 2
    dark_noise_sq = noise_params.dark_current_e_per_s * exposure_time_s
    
    return np.sqrt(read_noise_sq + dark_noise_sq)
