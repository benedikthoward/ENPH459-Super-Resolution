"""
Forward Model for Super Resolution Simulation

A physics-based simulation pipeline that models the complete optical chain from
scene to sensor output, including:

- Scene: SVG rendering to high-resolution raster with physical scale
- Optics: Lens PSF blur and motion blur modeling
- Optotune: Tilted plate beam shift with Snell's law field-dependent warp
- Sensor: Pixel integration, noise model, and ADC quantization
"""

from .pipeline.simulate_frame import simulate_frame, FrameSimulator
from .pipeline.simulate_sequence import simulate_sequence, SequenceSimulator

__all__ = [
    "simulate_frame",
    "simulate_sequence", 
    "FrameSimulator",
    "SequenceSimulator",
]
