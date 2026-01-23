#!/usr/bin/env python3
"""
Full pipeline validation script.

Generates a complete simulation and saves all intermediate outputs
for inspection and debugging.

Usage:
    python -m scripts.validate_full_pipeline --output-dir validation_output
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Validate full forward model pipeline")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Path to SVG scene (uses synthetic if not provided)",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Full Pipeline Validation")
    print("=" * 60)
    
    # Import forward model
    from forward_model.scene.scene_units import SceneUnits, PhysicalExtent
    from forward_model.optics.lens_model import ThinLensModel
    from forward_model.optics.psf_model import DiffractionPSF
    from forward_model.optics.motion_blur import MotionBlurModel
    from forward_model.optotune.plate_model import PlateModel
    from forward_model.optotune.warp_field import create_warp_field
    from forward_model.sensor.pixel_integration import PixelIntegrator
    from forward_model.sensor.noise_model import NoiseModel, SensorNoise
    from forward_model.sensor.adc_model import ADCModel
    from forward_model.pipeline.simulate_frame import FrameSimulator, DebugOutput
    
    # Configuration
    print("\n1. Setting up simulation parameters...")
    
    # Lens
    lens = ThinLensModel(
        focal_length_mm=25.0,
        object_distance_mm=100.0,
        f_number=2.8,
        wavelength_nm=550,
    )
    print(f"  Lens: f={lens.focal_length_mm}mm, f/{lens.f_number}")
    print(f"  Magnification: {lens.magnification():.3f}x")
    print(f"  Image distance: {lens.image_distance_mm():.2f}mm")
    
    # Scene
    sensor_width_px = 512
    sensor_height_px = 384
    pixel_pitch_um = 3.45
    oversampling = 4
    
    sensor_width_mm = sensor_width_px * pixel_pitch_um / 1000
    scene_width_mm = sensor_width_mm / lens.magnification()
    scene_height_mm = scene_width_mm * sensor_height_px / sensor_width_px
    
    scene_units = SceneUnits(
        physical_extent=PhysicalExtent(
            width_mm=scene_width_mm,
            height_mm=scene_height_mm,
        ),
        oversampling_factor=oversampling,
        sensor_pixel_pitch_um=pixel_pitch_um,
        magnification=lens.magnification(),
    )
    
    print(f"\n  Scene: {scene_width_mm:.2f} x {scene_height_mm:.2f} mm")
    print(f"  Internal resolution: {scene_units.internal_resolution}")
    print(f"  Sensor resolution: {scene_units.sensor_resolution}")
    
    # Create test scene
    print("\n2. Creating test scene...")
    
    if args.scene is not None:
        from forward_model.scene.svg_loader import load_svg
        scene_img = load_svg(args.scene, scene_units)
        print(f"  Loaded scene from: {args.scene}")
    else:
        scene_img = create_test_scene(scene_units.internal_resolution)
        print("  Created synthetic test pattern")
    
    save_image(scene_img, output_dir / "01_scene.png")
    
    # Apply PSF
    print("\n3. Applying PSF blur...")
    
    psf = DiffractionPSF(
        wavelength_nm=lens.wavelength_nm,
        f_number=lens.f_number,
    )
    pixel_pitch_um_internal = scene_units.internal_pixel_pitch_mm * 1000
    after_psf = psf.apply(scene_img, pixel_pitch_um_internal)
    
    print(f"  PSF FWHM: {psf.fwhm_um:.2f} μm")
    print(f"  Internal pixel pitch: {pixel_pitch_um_internal:.3f} μm")
    
    save_image(after_psf, output_dir / "02_after_psf.png")
    
    # Apply motion blur (optional)
    print("\n4. Applying motion blur...")
    
    motion = MotionBlurModel(
        velocity_mm_s=(5.0, 0.0),  # 5 mm/s in X
        exposure_time_s=0.001,  # 1ms exposure
        magnification=lens.magnification(),
    )
    
    motion_distance_px = motion.motion_distance_pixels(pixel_pitch_um)
    print(f"  Velocity: {motion.velocity_mm_s} mm/s")
    print(f"  Exposure: {motion.exposure_time_s*1000:.1f} ms")
    print(f"  Motion blur: {motion.motion_distance_object_mm*1000:.1f} μm ({motion_distance_px:.1f} sensor pixels)")
    
    after_motion = motion.apply(after_psf, pixel_pitch_um_internal)
    save_image(after_motion, output_dir / "03_after_motion.png")
    
    # Apply Optotune warp
    print("\n5. Applying Optotune shift...")
    
    optotune = PlateModel(
        thickness_mm=2.0,
        refractive_index=1.517,
        tilt_x_deg=0.0,
        tilt_y_deg=0.5,
    )
    
    warp = create_warp_field(optotune, scene_units, lens.image_distance_mm())
    after_warp = warp.apply(after_motion)
    
    print(f"  Tilt: ({optotune.tilt_x_deg}, {optotune.tilt_y_deg}) degrees")
    print(f"  Uniform shift: {optotune.uniform_shift()} mm")
    print(f"  Max shift in image: {warp.max_shift_px:.2f} pixels")
    
    save_image(after_warp, output_dir / "04_after_warp.png")
    
    # Save warp field visualization
    flow_vis = warp.to_flow_visualization()
    plt.imsave(str(output_dir / "04_warp_field.png"), flow_vis)
    
    # Pixel integration
    print("\n6. Pixel integration (downsampling)...")
    
    integrator = PixelIntegrator(oversampling_factor=oversampling)
    after_integration = integrator.integrate(after_warp)
    
    print(f"  Oversampling: {oversampling}x")
    print(f"  Output shape: {after_integration.shape}")
    
    save_image(after_integration, output_dir / "05_after_integration.png")
    
    # Apply noise
    print("\n7. Applying sensor noise...")
    
    noise_params = SensorNoise(
        read_noise_e=2.5,
        dark_current_e_per_s=0.5,
        quantum_efficiency=0.7,
    )
    noise_model = NoiseModel(
        noise_params=noise_params,
        exposure_time_s=0.001,
    )
    
    photons_at_saturation = 10000
    photons = after_integration * photons_at_saturation
    electrons = noise_model.apply(photons)
    
    print(f"  Read noise: {noise_params.read_noise_e} e-")
    print(f"  Dark current: {noise_params.dark_current_e_per_s} e-/s")
    print(f"  QE: {noise_params.quantum_efficiency}")
    
    save_image(electrons / electrons.max(), output_dir / "06_after_noise.png")
    
    # ADC
    print("\n8. Applying ADC conversion...")
    
    adc = ADCModel(
        bit_depth=12,
        gain=0.38,
        black_level_dn=64,
    )
    
    final_dn = adc.apply(electrons)
    
    print(f"  Bit depth: {adc.bit_depth}")
    print(f"  Black level: {adc.black_level_dn} DN")
    print(f"  Output range: {final_dn.min()} - {final_dn.max()} DN")
    
    save_image(final_dn / final_dn.max(), output_dir / "07_final_dn.png")
    
    # Save 16-bit TIFF
    import cv2
    cv2.imwrite(str(output_dir / "07_final_dn.tiff"), final_dn)
    
    # Create comparison figure
    print("\n9. Creating comparison figure...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(scene_img, cmap='gray')
    axes[0, 0].set_title('1. Scene')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(after_psf, cmap='gray')
    axes[0, 1].set_title('2. After PSF')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(after_motion, cmap='gray')
    axes[0, 2].set_title('3. After Motion Blur')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(after_warp, cmap='gray')
    axes[0, 3].set_title('4. After Optotune Warp')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(after_integration, cmap='gray')
    axes[1, 0].set_title('5. After Integration')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(electrons / electrons.max(), cmap='gray')
    axes[1, 1].set_title('6. After Noise')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(final_dn, cmap='gray')
    axes[1, 2].set_title('7. Final (DN)')
    axes[1, 2].axis('off')
    
    # Histogram
    axes[1, 3].hist(final_dn.flatten(), bins=100, color='blue', alpha=0.7)
    axes[1, 3].set_xlabel('DN')
    axes[1, 3].set_ylabel('Count')
    axes[1, 3].set_title('DN Histogram')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pipeline_overview.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'pipeline_overview.png'}")
    
    print("\n" + "=" * 60)
    print("Pipeline validation complete!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


def create_test_scene(resolution):
    """Create a synthetic test scene with various features."""
    h, w = resolution[1], resolution[0]
    scene = np.ones((h, w), dtype=np.float32) * 0.8
    
    # Vertical bars (varying frequency)
    bar_region_h = h // 4
    for i, freq in enumerate([4, 8, 16, 32]):
        y_start = i * bar_region_h
        y_end = (i + 1) * bar_region_h
        x = np.arange(w)
        pattern = (np.sin(2 * np.pi * freq * x / w) > 0).astype(np.float32)
        scene[y_start:y_end, :] = pattern * 0.6 + 0.2
    
    # Add some isolated points
    for _ in range(10):
        cx = np.random.randint(w // 4, 3 * w // 4)
        cy = np.random.randint(h // 4, 3 * h // 4)
        scene[max(0, cy-3):min(h, cy+4), max(0, cx-3):min(w, cx+4)] = 0.0
    
    return scene


def save_image(arr, path):
    """Save normalized image."""
    import cv2
    arr_norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-10) * 255)
    cv2.imwrite(str(path), arr_norm.astype(np.uint8))
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
