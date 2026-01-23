"""
Sensor modeling module.

Simulates the complete sensor chain from photons to digital output:
- Pixel integration (downsampling from oversampled representation)
- Noise model (shot noise, read noise, dark current)
- ADC (gain, black level, quantization)
"""

from .pixel_integration import PixelIntegrator, integrate_pixels
from .noise_model import NoiseModel, SensorNoise, apply_sensor_noise
from .adc_model import ADCModel, apply_adc

__all__ = [
    "PixelIntegrator",
    "integrate_pixels",
    "NoiseModel",
    "SensorNoise", 
    "apply_sensor_noise",
    "ADCModel",
    "apply_adc",
]
