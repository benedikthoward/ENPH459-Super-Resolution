#!/usr/bin/env python3
"""
Validation script for PSF models.

Generates plots to verify:
1. PSF shape matches diffraction theory
2. FWHM matches f/# and wavelength
3. Comparison of different PSF modes

Usage:
    python -m scripts.validate_psf_model --output-dir validation_output
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Validate PSF models")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_output",
        help="Output directory for plots",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from forward_model.optics.psf_model import DiffractionPSF, GaussianPSF
    from forward_model.optics.lens_model import ThinLensModel
    
    print("=" * 60)
    print("PSF Model Validation")
    print("=" * 60)
    
    # Test parameters
    wavelength_nm = 550
    f_number = 2.8
    pixel_pitch_um = 0.5  # Fine sampling for accuracy
    
    # Create PSF models
    diff_psf = DiffractionPSF(wavelength_nm=wavelength_nm, f_number=f_number)
    gauss_psf = GaussianPSF.from_lens(wavelength_nm, f_number)
    
    print(f"\nTest parameters:")
    print(f"  Wavelength: {wavelength_nm} nm")
    print(f"  F-number: f/{f_number}")
    print(f"  Pixel pitch: {pixel_pitch_um} μm")
    
    # Test 1: Generate and compare PSFs
    print("\n1. Generating PSF kernels...")
    
    diff_kernel = diff_psf.generate_kernel(pixel_pitch_um, size_px=101)
    gauss_kernel = gauss_psf.generate_kernel(pixel_pitch_um, size_px=101)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Diffraction PSF
    im0 = axes[0].imshow(diff_kernel, cmap='hot')
    axes[0].set_title('Diffraction-Limited PSF')
    plt.colorbar(im0, ax=axes[0])
    
    # Gaussian PSF
    im1 = axes[1].imshow(gauss_kernel, cmap='hot')
    axes[1].set_title('Gaussian PSF')
    plt.colorbar(im1, ax=axes[1])
    
    # Difference
    diff = diff_kernel - gauss_kernel
    im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-diff.max(), vmax=diff.max())
    axes[2].set_title('Difference (Diffraction - Gaussian)')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'psf_comparison.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'psf_comparison.png'}")
    
    # Test 2: Radial profiles
    print("\n2. Computing radial profiles...")
    
    center = diff_kernel.shape[0] // 2
    r_max = 30
    
    r_px = np.arange(r_max + 1)
    r_um = r_px * pixel_pitch_um
    
    # Extract radial profiles (average over angles)
    diff_profile = []
    gauss_profile = []
    
    for r in range(r_max + 1):
        mask = np.zeros_like(diff_kernel, dtype=bool)
        y, x = np.ogrid[:diff_kernel.shape[0], :diff_kernel.shape[1]]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        mask = (dist >= r - 0.5) & (dist < r + 0.5)
        
        if mask.any():
            diff_profile.append(diff_kernel[mask].mean())
            gauss_profile.append(gauss_kernel[mask].mean())
        else:
            diff_profile.append(0)
            gauss_profile.append(0)
    
    diff_profile = np.array(diff_profile)
    gauss_profile = np.array(gauss_profile)
    
    # Normalize
    diff_profile /= diff_profile.max()
    gauss_profile /= gauss_profile.max()
    
    # Theoretical Airy pattern
    # I(x) = [2*J1(x)/x]^2, where x = pi * r / (wavelength * f/#)
    from scipy.special import j1
    
    wavelength_um = wavelength_nm / 1000
    x_theory = np.pi * r_um / (wavelength_um * f_number)
    with np.errstate(divide='ignore', invalid='ignore'):
        airy = (2 * j1(x_theory) / x_theory)**2
    airy[0] = 1.0  # Fix singularity at r=0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r_um, diff_profile, 'b-', linewidth=2, label='Computed (Diffraction)')
    ax.plot(r_um, gauss_profile, 'g--', linewidth=2, label='Gaussian approx.')
    ax.plot(r_um, airy, 'r:', linewidth=2, label='Theoretical Airy')
    ax.set_xlabel('Radius (μm)', fontsize=12)
    ax.set_ylabel('Normalized Intensity', fontsize=12)
    ax.set_title('PSF Radial Profile', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, r_um.max())
    ax.set_ylim(0, 1.1)
    
    # Mark theoretical FWHM and Airy radius
    fwhm_um = diff_psf.fwhm_um
    airy_radius_um = diff_psf.airy_radius_um
    ax.axvline(x=fwhm_um/2, color='b', linestyle=':', alpha=0.5, label=f'FWHM/2 = {fwhm_um/2:.2f} μm')
    ax.axvline(x=airy_radius_um, color='r', linestyle=':', alpha=0.5, label=f'Airy radius = {airy_radius_um:.2f} μm')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'psf_radial_profile.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'psf_radial_profile.png'}")
    
    # Test 3: FWHM verification
    print("\n3. Verifying FWHM...")
    
    # Theoretical FWHM for Airy pattern: ~1.03 * lambda * f/#
    theoretical_fwhm = 1.03 * wavelength_um * f_number
    computed_fwhm = diff_psf.fwhm_um
    
    # Measure FWHM from computed profile
    half_max_idx = np.where(diff_profile <= 0.5)[0]
    if len(half_max_idx) > 0:
        measured_fwhm = 2 * r_um[half_max_idx[0]]
    else:
        measured_fwhm = 2 * r_um[-1]
    
    print(f"\n  Theoretical FWHM: {theoretical_fwhm:.3f} μm")
    print(f"  Model FWHM:       {computed_fwhm:.3f} μm")
    print(f"  Measured FWHM:    {measured_fwhm:.3f} μm")
    
    error_percent = abs(measured_fwhm - theoretical_fwhm) / theoretical_fwhm * 100
    if error_percent < 15:
        print(f"  ✓ FWHM matches theory within {error_percent:.1f}%")
    else:
        print(f"  ⚠ FWHM deviates from theory by {error_percent:.1f}%")
    
    # Test 4: Airy radius verification
    print("\n4. Verifying Airy radius (first zero)...")
    
    # Theoretical Airy radius: 1.22 * lambda * f/#
    theoretical_airy = 1.22 * wavelength_um * f_number
    computed_airy = diff_psf.airy_radius_um
    
    print(f"  Theoretical Airy radius: {theoretical_airy:.3f} μm")
    print(f"  Computed Airy radius:    {computed_airy:.3f} μm")
    
    # Test 5: Different f-numbers
    print("\n5. Testing FWHM vs. f-number relationship...")
    
    f_numbers = [1.4, 2.0, 2.8, 4.0, 5.6, 8.0]
    fwhm_values = []
    
    for fn in f_numbers:
        psf = DiffractionPSF(wavelength_nm=wavelength_nm, f_number=fn)
        fwhm_values.append(psf.fwhm_um)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(f_numbers, fwhm_values, 'bo-', linewidth=2, markersize=8, label='Computed')
    
    # Theoretical: FWHM = 1.03 * lambda * f/#
    f_theory = np.linspace(1, 9, 100)
    fwhm_theory = 1.03 * wavelength_um * f_theory
    ax.plot(f_theory, fwhm_theory, 'r--', linewidth=1.5, label='Theory (1.03λf/#)')
    
    ax.set_xlabel('F-number', fontsize=12)
    ax.set_ylabel('FWHM (μm)', fontsize=12)
    ax.set_title(f'PSF FWHM vs. F-number (λ = {wavelength_nm} nm)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fwhm_vs_fnumber.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'fwhm_vs_fnumber.png'}")
    
    print("\n" + "=" * 60)
    print("PSF validation complete! Check output directory for plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
