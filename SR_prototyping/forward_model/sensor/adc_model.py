"""
Analog-to-Digital Converter (ADC) modeling.

Converts electron counts to digital numbers (DN) with:
- Gain (electrons per DN)
- Black level offset
- Quantization to integer bit depth
- Saturation clipping
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ADCModel:
    """
    ADC model for converting electrons to digital numbers.
    
    The conversion follows:
    DN = clip((electrons * gain) + black_level, 0, 2^bits - 1)
    
    Attributes:
        bit_depth: Number of bits (e.g., 8, 10, 12, 14, 16)
        gain: Conversion gain in DN/electron (inverse of electrons per DN)
        black_level_dn: Black level offset in DN
        full_well_e: Full well capacity in electrons (for reference)
    """
    bit_depth: int = 12
    gain: float = 1.0
    black_level_dn: int = 64
    full_well_e: float = 10000
    
    def __post_init__(self):
        self._max_dn = (1 << self.bit_depth) - 1
        self._electrons_per_dn = 1.0 / self.gain if self.gain > 0 else 1.0
    
    @property
    def max_dn(self) -> int:
        """Maximum digital number value."""
        return self._max_dn
    
    @property
    def electrons_per_dn(self) -> float:
        """Electrons required per DN step."""
        return self._electrons_per_dn
    
    @property
    def dynamic_range_db(self) -> float:
        """Dynamic range in decibels."""
        usable_dn = self._max_dn - self.black_level_dn
        return 20 * np.log10(usable_dn)
    
    def apply(
        self,
        electrons: np.ndarray,
        quantize: bool = True,
    ) -> np.ndarray:
        """
        Convert electron counts to digital numbers.
        
        Args:
            electrons: Input electron counts
            quantize: If True, round to integers. If False, return float.
        
        Returns:
            Digital numbers (uint16 if quantized, float32 otherwise)
        """
        # Apply gain
        dn_float = electrons * self.gain + self.black_level_dn
        
        # Clip to valid range
        dn_clipped = np.clip(dn_float, 0, self._max_dn)
        
        if quantize:
            # Round and convert to appropriate integer type
            dn_int = np.round(dn_clipped).astype(np.uint16)
            return dn_int
        else:
            return dn_clipped.astype(np.float32)
    
    def inverse(
        self,
        dn: np.ndarray,
    ) -> np.ndarray:
        """
        Convert digital numbers back to electrons (approximate).
        
        Args:
            dn: Digital number values
        
        Returns:
            Estimated electron counts
        """
        return (dn.astype(np.float32) - self.black_level_dn) * self._electrons_per_dn
    
    def saturation_mask(
        self,
        dn: np.ndarray,
        threshold: float = 0.95,
    ) -> np.ndarray:
        """
        Create a mask of saturated pixels.
        
        Args:
            dn: Digital number image
            threshold: Fraction of max_dn to consider saturated
        
        Returns:
            Boolean mask (True where saturated)
        """
        sat_level = int(threshold * self._max_dn)
        return dn >= sat_level
    
    @classmethod
    def from_full_scale(
        cls,
        bit_depth: int,
        full_well_e: float,
        black_level_dn: int = 0,
    ) -> 'ADCModel':
        """
        Create ADC model calibrated to map full well to full scale.
        
        Args:
            bit_depth: Number of bits
            full_well_e: Full well capacity
            black_level_dn: Black level offset
        
        Returns:
            ADCModel instance
        """
        max_dn = (1 << bit_depth) - 1
        usable_dn = max_dn - black_level_dn
        gain = usable_dn / full_well_e
        
        return cls(
            bit_depth=bit_depth,
            gain=gain,
            black_level_dn=black_level_dn,
            full_well_e=full_well_e,
        )


def apply_adc(
    electrons: np.ndarray,
    bit_depth: int = 12,
    gain: float = 1.0,
    black_level_dn: int = 64,
    quantize: bool = True,
) -> np.ndarray:
    """
    Convenience function to apply ADC conversion.
    
    Args:
        electrons: Input electron counts
        bit_depth: ADC bit depth
        gain: Conversion gain (DN/electron)
        black_level_dn: Black level offset
        quantize: Whether to quantize to integers
    
    Returns:
        Digital number output
    """
    adc = ADCModel(
        bit_depth=bit_depth,
        gain=gain,
        black_level_dn=black_level_dn,
    )
    return adc.apply(electrons, quantize=quantize)


def compute_optimal_gain(
    full_well_e: float,
    bit_depth: int,
    black_level_dn: int = 0,
) -> float:
    """
    Compute optimal gain to map full well to full scale.
    
    Args:
        full_well_e: Full well capacity in electrons
        bit_depth: ADC bit depth
        black_level_dn: Black level offset
    
    Returns:
        Optimal gain value (DN/electron)
    """
    max_dn = (1 << bit_depth) - 1
    usable_dn = max_dn - black_level_dn
    return usable_dn / full_well_e


def dn_to_normalized(
    dn: np.ndarray,
    bit_depth: int = 12,
    black_level_dn: int = 64,
) -> np.ndarray:
    """
    Convert DN to normalized [0, 1] range.
    
    Args:
        dn: Digital number image
        bit_depth: ADC bit depth
        black_level_dn: Black level offset
    
    Returns:
        Normalized image (0-1 range)
    """
    max_dn = (1 << bit_depth) - 1
    usable_range = max_dn - black_level_dn
    
    normalized = (dn.astype(np.float32) - black_level_dn) / usable_range
    return np.clip(normalized, 0, 1)


def normalized_to_dn(
    normalized: np.ndarray,
    bit_depth: int = 12,
    black_level_dn: int = 64,
    quantize: bool = True,
) -> np.ndarray:
    """
    Convert normalized [0, 1] to DN.
    
    Args:
        normalized: Normalized image (0-1 range)
        bit_depth: ADC bit depth
        black_level_dn: Black level offset
        quantize: Whether to quantize to integers
    
    Returns:
        Digital number image
    """
    max_dn = (1 << bit_depth) - 1
    usable_range = max_dn - black_level_dn
    
    dn_float = normalized * usable_range + black_level_dn
    dn_clipped = np.clip(dn_float, 0, max_dn)
    
    if quantize:
        return np.round(dn_clipped).astype(np.uint16)
    return dn_clipped.astype(np.float32)
