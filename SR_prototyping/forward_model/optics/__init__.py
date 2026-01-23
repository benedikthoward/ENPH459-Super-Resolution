"""
Optical modeling module.

Includes lens geometry, PSF generation (diffraction-limited and aberrated),
and motion blur modeling.
"""

from .lens_model import LensModel, ThinLensModel
from .psf_model import PSFModel, DiffractionPSF, GaussianPSF
from .motion_blur import MotionBlurModel, apply_motion_blur

__all__ = [
    "LensModel",
    "ThinLensModel",
    "PSFModel",
    "DiffractionPSF",
    "GaussianPSF",
    "MotionBlurModel",
    "apply_motion_blur",
]
