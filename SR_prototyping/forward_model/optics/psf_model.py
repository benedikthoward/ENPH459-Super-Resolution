"""
Point Spread Function (PSF) modeling.

Supports diffraction-limited PSF, Gaussian approximation, and Zernike-based
aberrated PSF generation using prysm for accurate physical optics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve


class PSFModel(ABC):
    """Abstract base class for PSF models."""
    
    @abstractmethod
    def generate_kernel(
        self,
        pixel_pitch_um: float,
        size_px: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate the PSF kernel at the specified resolution.
        
        Args:
            pixel_pitch_um: Size of pixels in the kernel (micrometers)
            size_px: Size of the kernel (odd integer). If None, auto-computed.
        
        Returns:
            2D PSF kernel normalized to sum to 1
        """
        pass
    
    def apply(
        self,
        image: np.ndarray,
        pixel_pitch_um: float,
    ) -> np.ndarray:
        """
        Apply the PSF blur to an image.
        
        Args:
            image: Input image (2D or 3D for color)
            pixel_pitch_um: Pixel size of the input image
        
        Returns:
            Blurred image
        """
        kernel = self.generate_kernel(pixel_pitch_um)
        
        if image.ndim == 2:
            return fftconvolve(image, kernel, mode='same')
        else:
            # Handle color images
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = fftconvolve(image[:, :, c], kernel, mode='same')
            return result


@dataclass
class DiffractionPSF(PSFModel):
    """
    Diffraction-limited PSF based on Airy pattern.
    
    The Airy pattern intensity is:
    I(r) = I_0 * [2 * J_1(x) / x]^2
    where x = pi * D * r / (wavelength * f)
    
    Attributes:
        wavelength_nm: Wavelength of light in nanometers
        f_number: F-number of the lens
        defocus_waves: Defocus aberration in waves (optional)
    """
    wavelength_nm: float = 550.0
    f_number: float = 2.8
    defocus_waves: float = 0.0
    
    @property
    def airy_radius_um(self) -> float:
        """Radius to first zero of Airy pattern in micrometers."""
        return 1.22 * (self.wavelength_nm / 1000.0) * self.f_number
    
    @property
    def fwhm_um(self) -> float:
        """Full width at half maximum of the Airy disk in micrometers."""
        # FWHM ≈ 1.03 * wavelength * f_number
        return 1.03 * (self.wavelength_nm / 1000.0) * self.f_number
    
    def generate_kernel(
        self,
        pixel_pitch_um: float,
        size_px: Optional[int] = None,
    ) -> np.ndarray:
        """Generate diffraction-limited PSF using prysm if available."""
        try:
            return self._generate_with_prysm(pixel_pitch_um, size_px)
        except ImportError:
            # Fall back to analytical Airy pattern
            return self._generate_analytical(pixel_pitch_um, size_px)
    
    def _generate_with_prysm(
        self,
        pixel_pitch_um: float,
        size_px: Optional[int] = None,
    ) -> np.ndarray:
        """Generate PSF using prysm library for accurate physical optics."""
        from prysm.coordinates import make_xy_grid
        from prysm.geometry import circle
        from prysm.propagation import Wavefront
        
        # Determine kernel size
        if size_px is None:
            # Make kernel large enough to capture most of the energy
            extent_um = 6 * self.airy_radius_um
            size_px = int(np.ceil(extent_um / pixel_pitch_um))
            size_px = size_px | 1  # Ensure odd
        
        # Set up the pupil
        wavelength_um = self.wavelength_nm / 1000.0
        aperture_um = wavelength_um * self.f_number * size_px / 2
        
        x, y = make_xy_grid(size_px, diameter=aperture_um)
        r = np.sqrt(x**2 + y**2)
        
        # Create circular pupil
        pupil = circle(aperture_um / 2, r)
        
        # Add defocus if specified
        if self.defocus_waves != 0:
            phase = self.defocus_waves * 2 * np.pi * (2 * (r / (aperture_um/2))**2 - 1)
            phase = np.where(pupil > 0, phase, 0)
            pupil = pupil * np.exp(1j * phase)
        
        # Propagate to focal plane
        wf = Wavefront(pupil, wavelength_um, pixel_pitch_um)
        psf_field = wf.focus(self.f_number * 1000)  # prysm wants mm
        
        psf = np.abs(psf_field.data)**2
        psf = psf / psf.sum()
        
        return psf.astype(np.float32)
    
    def _generate_analytical(
        self,
        pixel_pitch_um: float,
        size_px: Optional[int] = None,
    ) -> np.ndarray:
        """Generate PSF using analytical Airy pattern formula."""
        from scipy.special import j1
        
        if size_px is None:
            extent_um = 6 * self.airy_radius_um
            size_px = int(np.ceil(extent_um / pixel_pitch_um))
            size_px = size_px | 1
        
        # Create coordinate grid
        center = size_px // 2
        y, x = np.ogrid[:size_px, :size_px]
        r_px = np.sqrt((x - center)**2 + (y - center)**2)
        r_um = r_px * pixel_pitch_um
        
        # Compute Airy pattern: I = [2*J1(x)/x]^2
        # x = pi * r / (wavelength * f_number)
        wavelength_um = self.wavelength_nm / 1000.0
        x_arg = np.pi * r_um / (wavelength_um * self.f_number)
        
        # Handle the singularity at r=0
        with np.errstate(divide='ignore', invalid='ignore'):
            airy = (2 * j1(x_arg) / x_arg)**2
        airy[r_px == 0] = 1.0
        
        # Normalize
        airy = airy / airy.sum()
        
        return airy.astype(np.float32)


@dataclass
class GaussianPSF(PSFModel):
    """
    Gaussian approximation to the PSF.
    
    Useful for quick simulations or when the exact PSF shape is less critical.
    
    Attributes:
        sigma_um: Standard deviation of the Gaussian in micrometers
    """
    sigma_um: float
    
    @classmethod
    def from_fwhm(cls, fwhm_um: float) -> "GaussianPSF":
        """Create a Gaussian PSF from full width at half maximum."""
        sigma = fwhm_um / (2 * np.sqrt(2 * np.log(2)))
        return cls(sigma_um=sigma)
    
    @classmethod
    def from_lens(cls, wavelength_nm: float, f_number: float) -> "GaussianPSF":
        """Create a Gaussian PSF matching diffraction-limited spot size."""
        # FWHM of Airy ≈ 1.03 * lambda * f/#
        fwhm_um = 1.03 * (wavelength_nm / 1000.0) * f_number
        return cls.from_fwhm(fwhm_um)
    
    @property
    def fwhm_um(self) -> float:
        """Full width at half maximum in micrometers."""
        return self.sigma_um * 2 * np.sqrt(2 * np.log(2))
    
    def generate_kernel(
        self,
        pixel_pitch_um: float,
        size_px: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Gaussian PSF kernel."""
        sigma_px = self.sigma_um / pixel_pitch_um
        
        if size_px is None:
            # 6 sigma captures 99.7% of energy
            size_px = int(np.ceil(6 * sigma_px))
            size_px = size_px | 1  # Ensure odd
        
        # Ensure minimum size
        size_px = max(size_px, 3)
        
        center = size_px // 2
        y, x = np.ogrid[:size_px, :size_px]
        r2 = (x - center)**2 + (y - center)**2
        
        kernel = np.exp(-r2 / (2 * sigma_px**2))
        kernel = kernel / kernel.sum()
        
        return kernel.astype(np.float32)


@dataclass  
class ZernikePSF(PSFModel):
    """
    PSF with Zernike polynomial aberrations.
    
    Uses prysm library for accurate wavefront propagation with aberrations.
    
    Attributes:
        wavelength_nm: Wavelength of light
        f_number: F-number of the lens
        zernike_coeffs: Dictionary of Zernike coefficients in waves
                       Keys are Noll indices (e.g., 4=defocus, 5,6=astigmatism)
    """
    wavelength_nm: float = 550.0
    f_number: float = 2.8
    zernike_coeffs: Dict[int, float] = field(default_factory=dict)
    
    def generate_kernel(
        self,
        pixel_pitch_um: float,
        size_px: Optional[int] = None,
    ) -> np.ndarray:
        """Generate aberrated PSF using Zernike polynomials."""
        try:
            from prysm.coordinates import make_xy_grid
            from prysm.geometry import circle
            from prysm.polynomials import zernike_nm_sequence, nm_to_fringe
            from prysm.propagation import Wavefront
        except ImportError:
            raise ImportError(
                "prysm library required for Zernike PSF generation. "
                "Install with: pip install prysm"
            )
        
        # Determine kernel size
        airy_radius_um = 1.22 * (self.wavelength_nm / 1000.0) * self.f_number
        if size_px is None:
            extent_um = 8 * airy_radius_um
            size_px = int(np.ceil(extent_um / pixel_pitch_um))
            size_px = size_px | 1
        
        wavelength_um = self.wavelength_nm / 1000.0
        aperture_um = wavelength_um * self.f_number * size_px / 2
        
        x, y = make_xy_grid(size_px, diameter=aperture_um)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Normalized radius
        rho = r / (aperture_um / 2)
        
        # Create pupil with aberrations
        pupil_mask = circle(aperture_um / 2, r)
        phase = np.zeros_like(r)
        
        # Add Zernike aberrations
        for noll_idx, coeff_waves in self.zernike_coeffs.items():
            if coeff_waves != 0:
                # Convert Noll to (n, m)
                n, m = self._noll_to_nm(noll_idx)
                zern = self._zernike(n, m, rho, theta)
                phase += coeff_waves * 2 * np.pi * zern
        
        phase = np.where(pupil_mask > 0, phase, 0)
        pupil = pupil_mask * np.exp(1j * phase)
        
        # Propagate
        wf = Wavefront(pupil, wavelength_um, pixel_pitch_um)
        psf_field = wf.focus(self.f_number * 1000)
        
        psf = np.abs(psf_field.data)**2
        psf = psf / psf.sum()
        
        return psf.astype(np.float32)
    
    @staticmethod
    def _noll_to_nm(noll: int) -> Tuple[int, int]:
        """Convert Noll index to (n, m) indices."""
        n = int(np.ceil((-3 + np.sqrt(9 + 8 * (noll - 1))) / 2))
        m_options = list(range(-n, n + 1, 2))
        
        # Find correct m
        idx = noll - (n * (n + 1)) // 2 - 1
        if n % 2 == 0:
            m = m_options[idx] if idx < len(m_options) else 0
        else:
            m = m_options[idx] if idx < len(m_options) else 0
            
        return n, m
    
    @staticmethod
    def _zernike(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute Zernike polynomial Z_n^m."""
        from scipy.special import factorial
        
        # Radial polynomial
        R = np.zeros_like(rho)
        abs_m = abs(m)
        
        for k in range((n - abs_m) // 2 + 1):
            coef = ((-1)**k * factorial(n - k) / 
                   (factorial(k) * factorial((n + abs_m)//2 - k) * 
                    factorial((n - abs_m)//2 - k)))
            R += coef * rho**(n - 2*k)
        
        # Apply azimuthal component
        if m >= 0:
            return R * np.cos(m * theta)
        else:
            return R * np.sin(-m * theta)


def apply_psf(
    image: np.ndarray,
    psf_model: PSFModel,
    pixel_pitch_um: float,
) -> np.ndarray:
    """
    Convenience function to apply a PSF model to an image.
    
    Args:
        image: Input image
        psf_model: PSF model instance
        pixel_pitch_um: Pixel pitch of the image
    
    Returns:
        Blurred image
    """
    return psf_model.apply(image, pixel_pitch_um)
