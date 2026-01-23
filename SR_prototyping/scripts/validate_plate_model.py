#!/usr/bin/env python3
"""
Validation script for the Optotune tilted plate model.

Generates plots to verify:
1. Shift vs. angle relationship follows Snell's law
2. Field-dependent warp shows expected variation
3. Computed shifts match expected values for XRP-20 parameters

Usage:
    python -m scripts.validate_plate_model --output-dir validation_output
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Validate Optotune plate model")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_output",
        help="Output directory for plots",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from forward_model.optotune.plate_model import (
        PlateModel, compute_plate_shift, shift_vs_angle_curve
    )
    
    print("=" * 60)
    print("Optotune Plate Model Validation")
    print("=" * 60)
    
    # Test 1: Shift vs. angle curve
    print("\n1. Validating shift vs. angle relationship...")
    
    plate = PlateModel(
        thickness_mm=2.0,
        refractive_index=1.517,  # BK7 glass
    )
    
    angles_deg, shifts_mm = shift_vs_angle_curve(plate, (-5, 5), 200)
    
    # Theoretical small-angle approximation
    # Delta ≈ t * theta * (n-1)/n for small theta
    angles_rad = np.deg2rad(angles_deg)
    approx_shifts = plate.thickness_mm * angles_rad * (plate.refractive_index - 1) / plate.refractive_index
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(angles_deg, shifts_mm, 'b-', linewidth=2, label='Exact (Snell\'s law)')
    ax.plot(angles_deg, approx_shifts, 'r--', linewidth=1.5, label='Small angle approx.')
    ax.set_xlabel('Tilt Angle (degrees)', fontsize=12)
    ax.set_ylabel('Lateral Shift (mm)', fontsize=12)
    ax.set_title('Optotune Plate Shift vs. Tilt Angle', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'shift_vs_angle.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'shift_vs_angle.png'}")
    
    # Print some reference values
    print("\n  Reference shifts at various angles:")
    for angle in [0.5, 1.0, 2.0, 5.0]:
        shift = compute_plate_shift(np.deg2rad(angle), plate.thickness_mm, plate.refractive_index)
        print(f"    {angle:5.1f}° -> {shift:.4f} mm ({shift*1000:.1f} μm)")
    
    # Test 2: Field-dependent shift
    print("\n2. Validating field-dependent shift...")
    
    plate.tilt_x_deg = 0.0
    plate.tilt_y_deg = 0.5  # Apply Y tilt
    
    # Create field grid (image space)
    image_distance_mm = 33.0  # Typical for 25mm lens at 100mm object distance
    x = np.linspace(-5, 5, 100)  # mm from optical axis
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    dx_mm, dy_mm = plate.shift_field(X, Y, image_distance_mm)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # X shift map
    im0 = axes[0].imshow(dx_mm, extent=[-5, 5, -5, 5], origin='lower', cmap='RdBu_r')
    axes[0].set_title('X Shift (mm)')
    axes[0].set_xlabel('X position (mm)')
    axes[0].set_ylabel('Y position (mm)')
    plt.colorbar(im0, ax=axes[0])
    
    # Y shift map
    im1 = axes[1].imshow(dy_mm, extent=[-5, 5, -5, 5], origin='lower', cmap='RdBu_r')
    axes[1].set_title('Y Shift (mm)')
    axes[1].set_xlabel('X position (mm)')
    axes[1].set_ylabel('Y position (mm)')
    plt.colorbar(im1, ax=axes[1])
    
    # Magnitude
    mag_mm = np.sqrt(dx_mm**2 + dy_mm**2)
    im2 = axes[2].imshow(mag_mm, extent=[-5, 5, -5, 5], origin='lower', cmap='viridis')
    axes[2].set_title('Shift Magnitude (mm)')
    axes[2].set_xlabel('X position (mm)')
    axes[2].set_ylabel('Y position (mm)')
    plt.colorbar(im2, ax=axes[2])
    
    plt.suptitle(f'Field-Dependent Shift (tilt_y = {plate.tilt_y_deg}°)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'field_dependent_shift.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'field_dependent_shift.png'}")
    
    # Print statistics
    print(f"\n  Field-dependent shift statistics:")
    print(f"    X shift range: {dx_mm.min():.4f} to {dx_mm.max():.4f} mm")
    print(f"    Y shift range: {dy_mm.min():.4f} to {dy_mm.max():.4f} mm")
    print(f"    Center shift: ({dx_mm[50, 50]:.4f}, {dy_mm[50, 50]:.4f}) mm")
    print(f"    Corner shift: ({dx_mm[0, 0]:.4f}, {dy_mm[0, 0]:.4f}) mm")
    
    # Test 3: Vector field visualization
    print("\n3. Generating vector field visualization...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Subsample for quiver plot
    skip = 5
    ax.quiver(
        X[::skip, ::skip], Y[::skip, ::skip],
        dx_mm[::skip, ::skip], dy_mm[::skip, ::skip],
        mag_mm[::skip, ::skip], cmap='viridis', scale=0.5
    )
    ax.set_xlabel('X position (mm)')
    ax.set_ylabel('Y position (mm)')
    ax.set_title(f'Shift Vector Field (tilt_y = {plate.tilt_y_deg}°)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'shift_vector_field.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'shift_vector_field.png'}")
    
    # Test 4: Compare with XRP-20 specs
    print("\n4. Comparing with XRP-20 specifications...")
    
    # From datasheet: approximately 0.68mm per degree at small angles
    expected_shift_per_deg = 0.68  # mm
    computed_shift_per_deg = compute_plate_shift(
        np.deg2rad(1.0), plate.thickness_mm, plate.refractive_index
    )
    
    error_percent = abs(computed_shift_per_deg - expected_shift_per_deg) / expected_shift_per_deg * 100
    
    print(f"  Expected shift at 1°: ~{expected_shift_per_deg:.2f} mm")
    print(f"  Computed shift at 1°:  {computed_shift_per_deg:.4f} mm")
    print(f"  Error: {error_percent:.1f}%")
    
    if error_percent < 10:
        print("  ✓ Model matches specifications within 10%")
    else:
        print("  ⚠ Model deviates from specifications by more than 10%")
    
    print("\n" + "=" * 60)
    print("Validation complete! Check output directory for plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
