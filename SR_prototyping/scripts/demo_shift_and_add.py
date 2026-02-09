#!/usr/bin/env python3
"""
Shift-and-Add Super Resolution Demo

Two modes:
  (A) TRANSLATION MODE: Synthetic HR truth → subpixel shifts → PSF blur → area downsample → LR
      Reconstruction uses known shifts to splat LR samples into HR grid.
  (B) WARP MODE: Uses Optotune warp forward model + forward-warp splatting for reconstruction.

Usage:
    export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
    python -m scripts.demo_shift_and_add --sr-mode translation
    python -m scripts.demo_shift_and_add --sr-mode warp
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2


# =============================================================================
# TRANSLATION MODE: Forward Model
# =============================================================================

def load_hr_truth(scene_path: Path, hr_resolution: Tuple[int, int]) -> np.ndarray:
    """
    Load scene image and resize to HR truth resolution.
    
    Args:
        scene_path: Path to scene image (SVG, PNG, etc.)
        hr_resolution: (width, height) of HR truth
    
    Returns:
        HR truth image as float32 in [0, 1]
    """
    if scene_path.suffix.lower() == '.svg':
        import cairosvg
        import io
        from PIL import Image
        
        # Render SVG at HR resolution
        png_data = cairosvg.svg2png(
            url=str(scene_path),
            output_width=hr_resolution[0],
            output_height=hr_resolution[1],
        )
        img = Image.open(io.BytesIO(png_data))
        hr_truth = np.array(img.convert('L')).astype(np.float32) / 255.0
    else:
        img = cv2.imread(str(scene_path), cv2.IMREAD_GRAYSCALE)
        hr_truth = cv2.resize(img, hr_resolution, interpolation=cv2.INTER_LANCZOS4)
        hr_truth = hr_truth.astype(np.float32) / 255.0
    
    return hr_truth


def create_gaussian_psf(sigma_px: float, size: int = None) -> np.ndarray:
    """Create a normalized Gaussian PSF kernel."""
    if size is None:
        size = int(6 * sigma_px) | 1  # Ensure odd
    size = max(size, 3)
    
    center = size // 2
    y, x = np.mgrid[:size, :size]
    psf = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma_px**2))
    return psf / psf.sum()


def generate_translation_frames(
    hr_truth: np.ndarray,
    n_frames: int,
    upscale_factor: int,
    psf_sigma_hr_px: float = 1.0,
    noise_std: float = 0.01,
    quantize: bool = True,
    bit_depth: int = 12,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]], np.ndarray]:
    """
    Generate LR frames from HR truth using translation + blur + area downsample.
    
    For 2x2 SR with upscale=2: shifts are (0,0), (1,0), (0,1), (1,1) in HR pixels
    For 4x4 SR with upscale=4: shifts are (0,0), (1,0), ..., (3,3) in HR pixels
    
    Args:
        hr_truth: HR truth image (float32, [0,1])
        n_frames: Number of frames (should be perfect square: 4, 9, 16, ...)
        upscale_factor: SR upscale factor (2, 3, 4, ...)
        psf_sigma_hr_px: PSF blur sigma in HR pixels
        noise_std: Gaussian noise standard deviation
        quantize: Whether to simulate ADC quantization
        bit_depth: ADC bit depth for quantization
    
    Returns:
        (frames, shifts, hr_blurred)
        - frames: List of LR frames (float32)
        - shifts: List of (dx, dy) shifts in HR pixels used for each frame
        - hr_blurred: HR truth after PSF blur (for reference)
    """
    grid_size = int(np.sqrt(n_frames))
    if grid_size * grid_size != n_frames:
        raise ValueError(f"n_frames={n_frames} must be a perfect square")
    
    if grid_size > upscale_factor:
        print(f"  Warning: grid_size={grid_size} > upscale_factor={upscale_factor}")
        print(f"  Some frames will have overlapping subpixel positions")
    
    h_hr, w_hr = hr_truth.shape
    h_lr, w_lr = h_hr // upscale_factor, w_hr // upscale_factor
    
    # Create PSF and blur HR truth
    psf = create_gaussian_psf(psf_sigma_hr_px)
    hr_blurred = cv2.filter2D(hr_truth, -1, psf, borderType=cv2.BORDER_REFLECT)
    
    print(f"\n  Generating {n_frames} LR frames ({grid_size}x{grid_size} grid)")
    print(f"  HR resolution: {w_hr} x {h_hr}")
    print(f"  LR resolution: {w_lr} x {h_lr}")
    print(f"  PSF sigma: {psf_sigma_hr_px:.2f} HR pixels")
    
    frames = []
    shifts = []
    
    for j in range(grid_size):
        for i in range(grid_size):
            # Subpixel shift in HR pixels
            # For upscale=4, grid=4: shifts = 0,1,2,3
            # For upscale=2, grid=2: shifts = 0,1
            dx = i % upscale_factor
            dy = j % upscale_factor
            shifts.append((dx, dy))
            
            # Shift the blurred HR image
            # Use affine transform for sub-pixel accuracy
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            hr_shifted = cv2.warpAffine(
                hr_blurred, M, (w_hr, h_hr),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT,
            )
            
            # Area downsample (pixel integration)
            lr_frame = cv2.resize(
                hr_shifted, (w_lr, h_lr),
                interpolation=cv2.INTER_AREA,
            )
            
            # Add noise
            if noise_std > 0:
                noise = np.random.randn(h_lr, w_lr).astype(np.float32) * noise_std
                lr_frame = lr_frame + noise
            
            # Quantize (simulate ADC)
            if quantize:
                max_val = 2**bit_depth - 1
                lr_frame = np.clip(lr_frame, 0, 1)
                lr_frame = np.round(lr_frame * max_val) / max_val
            
            lr_frame = np.clip(lr_frame, 0, 1).astype(np.float32)
            frames.append(lr_frame)
            
            print(f"    Frame {len(frames)-1:2d}: shift=({dx}, {dy}) HR px")
    
    return frames, shifts, hr_blurred


# =============================================================================
# WARP MODE: Forward Model
# =============================================================================

def generate_warp_frames(
    scene_path: Path,
    output_dir: Path,
    lens_params: dict,
    n_frames: int,
    sensor_resolution: Tuple[int, int],
) -> Tuple[List[np.ndarray], List[Any], dict]:
    """
    Generate LR frames using the Optotune warp forward model.
    
    Returns:
        (frames, warp_fields, metadata)
    """
    from forward_model.pipeline.simulate_frame import FrameSimulator
    from forward_model.scene.scene_units import SceneUnits, PhysicalExtent
    from forward_model.optics.lens_model import ThinLensModel
    from forward_model.optics.psf_model import DiffractionPSF
    from forward_model.optotune.plate_model import PlateModel
    from forward_model.optotune.warp_field import create_warp_field
    from forward_model.sensor.noise_model import SensorNoise
    from forward_model.sensor.adc_model import ADCModel
    
    grid_size = int(np.sqrt(n_frames))
    if grid_size * grid_size != n_frames:
        raise ValueError(f"n_frames={n_frames} must be a perfect square")
    
    # Create lens model
    lens = ThinLensModel(
        focal_length_mm=lens_params['focal_length_mm'],
        object_distance_mm=lens_params['object_distance_mm'],
        f_number=lens_params['f_number'],
        wavelength_nm=lens_params.get('wavelength_nm', 550),
    )
    
    print(f"\n  Lens: {lens.focal_length_mm}mm f/{lens.f_number}")
    print(f"  Working distance: {lens.object_distance_mm}mm")
    print(f"  Magnification: {lens.magnification():.4f}")
    
    sensor_pixel_pitch_um = 3.45
    sensor_width_mm = sensor_resolution[0] * sensor_pixel_pitch_um / 1000
    scene_width_mm = sensor_width_mm / lens.magnification()
    aspect = sensor_resolution[1] / sensor_resolution[0]
    scene_height_mm = scene_width_mm * aspect
    
    scene_units = SceneUnits(
        physical_extent=PhysicalExtent(
            width_mm=scene_width_mm,
            height_mm=scene_height_mm,
        ),
        oversampling_factor=4,
        sensor_pixel_pitch_um=sensor_pixel_pitch_um,
        magnification=lens.magnification(),
    )
    
    psf = DiffractionPSF(
        wavelength_nm=lens.wavelength_nm,
        f_number=lens.f_number,
    )
    
    base_optotune = PlateModel(
        thickness_mm=2.0,
        refractive_index=1.517,
        tilt_x_deg=0,
        tilt_y_deg=0,
    )
    
    noise = SensorNoise(
        read_noise_e=2.0,
        dark_current_e_per_s=0.3,
        quantum_efficiency=0.7,
    )
    
    adc = ADCModel(
        bit_depth=12,
        gain=1.0,
        black_level_dn=64,
        full_well_e=10000,
    )
    
    frame_sim = FrameSimulator(
        scene_units=scene_units,
        lens=lens,
        psf=psf,
        optotune=base_optotune,
        noise=noise,
        adc=adc,
        exposure_time_s=0.001,
    )
    
    # Generate grid of tilts
    shift_amplitude_deg = lens_params.get('shift_amplitude_deg', 0.1)
    
    tilts = []
    for j in range(grid_size):
        for i in range(grid_size):
            tx = (i - (grid_size - 1) / 2) * shift_amplitude_deg / ((grid_size - 1) / 2) if grid_size > 1 else 0
            ty = (j - (grid_size - 1) / 2) * shift_amplitude_deg / ((grid_size - 1) / 2) if grid_size > 1 else 0
            tilts.append((tx, ty))
    
    print(f"\n  Generating {n_frames} frames with tilt amplitude {shift_amplitude_deg}deg")
    
    frames = []
    warp_fields = []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (tx, ty) in enumerate(tilts):
        plate = PlateModel(
            thickness_mm=2.0,
            refractive_index=1.517,
            tilt_x_deg=tx,
            tilt_y_deg=ty,
        )
        
        warp = create_warp_field(
            plate,
            scene_units,
            lens.image_distance_mm(),
            target_shape=(sensor_resolution[1], sensor_resolution[0]),
        )
        warp_fields.append(warp)
        
        frame_sim.optotune = plate
        frame = frame_sim.simulate(str(scene_path), debug=False)
        
        # Linearize: subtract black level
        frame = (frame.astype(np.float32) - adc.black_level_dn) / (2**adc.bit_depth - 1 - adc.black_level_dn)
        frame = np.clip(frame, 0, 1)
        
        frames.append(frame)
        
        print(f"    Frame {i:2d}: tilt=({tx:+.4f}, {ty:+.4f})deg, mean_shift={warp.mean_shift_px:.2f}px")
    
    metadata = {
        'adc_black_level': adc.black_level_dn,
        'adc_bit_depth': adc.bit_depth,
        'sensor_resolution': sensor_resolution,
    }
    
    return frames, warp_fields, metadata


# =============================================================================
# TRANSLATION MODE: Reconstruction with Bilinear Splatting
# =============================================================================

def reconstruct_sr_translation(
    frames: List[np.ndarray],
    shifts: List[Tuple[int, int]],
    upscale_factor: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct SR image from LR frames using known translation shifts.
    
    Uses bilinear splatting: each LR sample is distributed to 4 neighboring
    HR pixels based on its fractional position.
    
    For integer shifts (as in our grid), this simplifies to direct placement
    but we implement full bilinear for generality.
    
    Args:
        frames: List of LR frames
        shifts: List of (dx, dy) shifts in HR pixels
        upscale_factor: SR upscale factor
    
    Returns:
        (sr_image, weight_map)
    """
    h_lr, w_lr = frames[0].shape
    h_hr = h_lr * upscale_factor
    w_hr = w_lr * upscale_factor
    
    accumulator = np.zeros((h_hr, w_hr), dtype=np.float64)
    weights = np.zeros((h_hr, w_hr), dtype=np.float64)
    
    print(f"\n  Reconstructing SR via translation splatting...")
    print(f"  LR: {w_lr}x{h_lr} -> HR: {w_hr}x{h_hr}")
    
    for frame_idx, (frame, (dx, dy)) in enumerate(zip(frames, shifts)):
        # Each LR pixel (px, py) maps to HR position:
        #   hr_x = px * upscale_factor + dx
        #   hr_y = py * upscale_factor + dy
        
        for py in range(h_lr):
            for px in range(w_lr):
                # Target HR position (can be fractional for subpixel shifts)
                hr_x = px * upscale_factor + dx
                hr_y = py * upscale_factor + dy
                
                val = frame[py, px]
                
                # Bilinear splat to 4 neighbors
                x0, y0 = int(np.floor(hr_x)), int(np.floor(hr_y))
                x1, y1 = x0 + 1, y0 + 1
                
                fx = hr_x - x0
                fy = hr_y - y0
                
                # Weights for each corner
                w00 = (1 - fx) * (1 - fy)
                w10 = fx * (1 - fy)
                w01 = (1 - fx) * fy
                w11 = fx * fy
                
                # Splat to neighbors
                if 0 <= y0 < h_hr and 0 <= x0 < w_hr:
                    accumulator[y0, x0] += val * w00
                    weights[y0, x0] += w00
                if 0 <= y0 < h_hr and 0 <= x1 < w_hr:
                    accumulator[y0, x1] += val * w10
                    weights[y0, x1] += w10
                if 0 <= y1 < h_hr and 0 <= x0 < w_hr:
                    accumulator[y1, x0] += val * w01
                    weights[y1, x0] += w01
                if 0 <= y1 < h_hr and 0 <= x1 < w_hr:
                    accumulator[y1, x1] += val * w11
                    weights[y1, x1] += w11
        
        print(f"    Frame {frame_idx}: shift=({dx}, {dy})")
    
    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        sr_image = np.where(weights > 0, accumulator / weights, 0)
    
    # Statistics
    covered = weights > 0
    coverage = np.sum(covered) / covered.size * 100
    print(f"\n  Coverage: {coverage:.1f}%")
    print(f"  Weight stats: min={weights[covered].min():.2f}, mean={weights[covered].mean():.2f}, max={weights.max():.2f}")
    
    return sr_image.astype(np.float32), weights


# =============================================================================
# WARP MODE: Reconstruction with Forward-Warp Splatting
# =============================================================================

def reconstruct_sr_warp(
    frames: List[np.ndarray],
    warp_fields: List,
    upscale_factor: int,
    reference_frame_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct SR image from LR frames using warp fields (forward-warp splatting).
    
    For each LR pixel, compute where it maps in the HR reference grid using
    the warp field displacement, then splat it there.
    
    The warp field tells us: to get pixel at (x,y) in warped image, sample from (x-dx, y-dy).
    So if we have a sample at LR (px, py), in the reference frame it came from (px+dx, py+dy).
    We map this to HR by multiplying by upscale_factor.
    
    Args:
        frames: List of LR frames
        warp_fields: List of WarpField objects
        upscale_factor: SR upscale factor
        reference_frame_idx: Which frame to use as reference (0 = first frame with zero tilt)
    
    Returns:
        (sr_image, weight_map)
    """
    h_lr, w_lr = frames[0].shape
    h_hr = h_lr * upscale_factor
    w_hr = w_lr * upscale_factor
    
    accumulator = np.zeros((h_hr, w_hr), dtype=np.float64)
    weights = np.zeros((h_hr, w_hr), dtype=np.float64)
    
    # Get reference warp (to compute relative displacement)
    ref_warp = warp_fields[reference_frame_idx]
    
    print(f"\n  Reconstructing SR via warp-based splatting...")
    print(f"  Reference frame: {reference_frame_idx}")
    print(f"  LR: {w_lr}x{h_lr} -> HR: {w_hr}x{h_hr}")
    
    for frame_idx, (frame, warp) in enumerate(zip(frames, warp_fields)):
        # Compute relative displacement from reference
        # The warp field dx, dy tells us where samples come FROM
        # So sample at (x,y) in this frame corresponds to reference position (x + dx, y + dy)
        # relative to reference warp
        rel_dx = warp.dx - ref_warp.dx  # displacement relative to reference
        rel_dy = warp.dy - ref_warp.dy
        
        mean_shift = np.sqrt(rel_dx**2 + rel_dy**2).mean()
        print(f"    Frame {frame_idx}: mean relative shift = {mean_shift:.3f} LR px")
        
        for py in range(h_lr):
            for px in range(w_lr):
                # This LR sample corresponds to reference LR position (px + rel_dx, py + rel_dy)
                # In HR coordinates:
                ref_lr_x = px + rel_dx[py, px]
                ref_lr_y = py + rel_dy[py, px]
                
                hr_x = ref_lr_x * upscale_factor
                hr_y = ref_lr_y * upscale_factor
                
                val = frame[py, px]
                
                # Bilinear splat
                x0, y0 = int(np.floor(hr_x)), int(np.floor(hr_y))
                x1, y1 = x0 + 1, y0 + 1
                
                fx = hr_x - x0
                fy = hr_y - y0
                
                w00 = (1 - fx) * (1 - fy)
                w10 = fx * (1 - fy)
                w01 = (1 - fx) * fy
                w11 = fx * fy
                
                if 0 <= y0 < h_hr and 0 <= x0 < w_hr:
                    accumulator[y0, x0] += val * w00
                    weights[y0, x0] += w00
                if 0 <= y0 < h_hr and 0 <= x1 < w_hr:
                    accumulator[y0, x1] += val * w10
                    weights[y0, x1] += w10
                if 0 <= y1 < h_hr and 0 <= x0 < w_hr:
                    accumulator[y1, x0] += val * w01
                    weights[y1, x0] += w01
                if 0 <= y1 < h_hr and 0 <= x1 < w_hr:
                    accumulator[y1, x1] += val * w11
                    weights[y1, x1] += w11
    
    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        sr_image = np.where(weights > 0, accumulator / weights, 0)
    
    # Statistics
    covered = weights > 0
    coverage = np.sum(covered) / covered.size * 100
    print(f"\n  Coverage: {coverage:.1f}%")
    if np.any(covered):
        print(f"  Weight stats: min={weights[covered].min():.2f}, mean={weights[covered].mean():.2f}, max={weights.max():.2f}")
    
    return sr_image.astype(np.float32), weights


# =============================================================================
# IBP for Translation Mode
# =============================================================================

def ibp_translation(
    sr_estimate: np.ndarray,
    lr_frames: List[np.ndarray],
    shifts: List[Tuple[int, int]],
    upscale_factor: int,
    psf_sigma_hr_px: float,
    num_iterations: int = 10,
    step_size: float = 0.5,
) -> np.ndarray:
    """
    Iterative Back Projection for translation mode.
    
    Forward model: HR -> blur -> shift -> area downsample -> LR
    Back-projection: error in LR -> upsample -> inverse shift -> blur transpose -> HR update
    """
    print(f"\n  IBP refinement ({num_iterations} iterations)...")
    
    hr = sr_estimate.copy().astype(np.float64)
    h_hr, w_hr = hr.shape
    h_lr, w_lr = h_hr // upscale_factor, w_hr // upscale_factor
    
    psf = create_gaussian_psf(psf_sigma_hr_px)
    
    for iteration in range(num_iterations):
        error_accum = np.zeros_like(hr)
        weight_accum = np.zeros_like(hr)
        total_error = 0.0
        
        for frame_idx, (lr_frame, (dx, dy)) in enumerate(zip(lr_frames, shifts)):
            # Forward: blur -> shift -> downsample
            hr_blurred = cv2.filter2D(hr, -1, psf, borderType=cv2.BORDER_REFLECT)
            
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            hr_shifted = cv2.warpAffine(
                hr_blurred, M, (w_hr, h_hr),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT,
            )
            
            simulated_lr = cv2.resize(hr_shifted, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
            
            # Error
            error_lr = lr_frame.astype(np.float64) - simulated_lr
            total_error += np.mean(np.abs(error_lr))
            
            # Back-project: upsample -> inverse shift -> blur
            error_hr = cv2.resize(error_lr, (w_hr, h_hr), interpolation=cv2.INTER_NEAREST)
            
            M_inv = np.float32([[1, 0, dx], [0, 1, dy]])
            error_aligned = cv2.warpAffine(
                error_hr, M_inv, (w_hr, h_hr),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT,
            )
            
            error_bp = cv2.filter2D(error_aligned, -1, psf, borderType=cv2.BORDER_REFLECT)
            
            error_accum += error_bp
            weight_accum += 1.0
        
        avg_error = error_accum / weight_accum
        hr = hr + step_size * avg_error
        hr = np.clip(hr, 0, 1.5)
        
        print(f"    Iter {iteration+1:2d}: MAE = {total_error / len(lr_frames):.6f}")
    
    return hr.astype(np.float32)


# =============================================================================
# Visualization
# =============================================================================

def save_comparison_figure(
    lr_frame: np.ndarray,
    sr_image: np.ndarray,
    bicubic: np.ndarray,
    output_path: Path,
    hr_truth: Optional[np.ndarray] = None,
    sr_ibp: Optional[np.ndarray] = None,
    metrics: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Create and save comparison figure with optional PSNR/SSIM annotations."""
    import matplotlib.pyplot as plt
    
    images = [('LR (nearest)', cv2.resize(lr_frame, (sr_image.shape[1], sr_image.shape[0]), interpolation=cv2.INTER_NEAREST), None)]
    
    bicubic_label = 'Bicubic'
    if metrics and 'bicubic' in metrics:
        bicubic_label = f"Bicubic\nPSNR: {metrics['bicubic']['psnr']:.1f} dB"
    images.append((bicubic_label, bicubic, None))
    
    sr_label = 'Shift-and-Add SR'
    if metrics and 'sr' in metrics:
        sr_label = f"Shift-and-Add SR\nPSNR: {metrics['sr']['psnr']:.1f} dB (+{metrics['sr']['psnr'] - metrics['bicubic']['psnr']:.1f})"
    images.append((sr_label, sr_image, None))
    
    if sr_ibp is not None:
        ibp_label = 'SR + IBP'
        if metrics and 'sr_ibp' in metrics:
            ibp_label = f"SR + IBP\nPSNR: {metrics['sr_ibp']['psnr']:.1f} dB (+{metrics['sr_ibp']['psnr'] - metrics['bicubic']['psnr']:.1f})"
        images.append((ibp_label, sr_ibp, None))
    
    if hr_truth is not None:
        images.append(('HR Truth', hr_truth, None))
    
    n_cols = len(images)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    for ax, (title, img, _) in zip(axes, images):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Shift-and-Add Super Resolution', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison to {output_path}")


def save_zoomed_comparison(
    lr_frame: np.ndarray,
    sr_image: np.ndarray,
    bicubic: np.ndarray,
    output_path: Path,
    hr_truth: Optional[np.ndarray] = None,
    sr_ibp: Optional[np.ndarray] = None,
    crop_ratio: float = 0.3,
) -> None:
    """Create zoomed comparison on center crop."""
    import matplotlib.pyplot as plt
    
    h, w = sr_image.shape
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    cy, cx = h // 2, w // 2
    y1, y2 = cy - crop_h // 2, cy + crop_h // 2
    x1, x2 = cx - crop_w // 2, cx + crop_w // 2
    
    lr_up = cv2.resize(lr_frame, (w, h), interpolation=cv2.INTER_NEAREST)
    
    images = [('LR (nearest)', lr_up[y1:y2, x1:x2])]
    images.append(('Bicubic', bicubic[y1:y2, x1:x2]))
    images.append(('Shift-and-Add', sr_image[y1:y2, x1:x2]))
    
    if sr_ibp is not None:
        images.append(('SR + IBP', sr_ibp[y1:y2, x1:x2]))
    
    if hr_truth is not None:
        images.append(('HR Truth', hr_truth[y1:y2, x1:x2]))
    
    n_cols = len(images)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    for ax, (title, img) in zip(axes, images):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Zoomed Comparison (Center Crop)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved zoomed comparison to {output_path}")


def compute_metrics(sr_image: np.ndarray, hr_truth: np.ndarray) -> Dict[str, float]:
    """Compute PSNR and SSIM between SR and HR truth."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    
    psnr = peak_signal_noise_ratio(hr_truth, sr_image, data_range=1.0)
    ssim = structural_similarity(hr_truth, sr_image, data_range=1.0)
    
    return {'psnr': psnr, 'ssim': ssim}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Shift-and-Add Super Resolution Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Mode selection
    parser.add_argument("--sr-mode", type=str, choices=['translation', 'warp'], default='translation',
                        help="SR mode: 'translation' uses synthetic shifts, 'warp' uses Optotune model")
    
    # Frame settings
    parser.add_argument("--n-frames", type=int, default=4,
                        help="Number of frames (must be perfect square: 4, 9, 16, ...)")
    parser.add_argument("--upscale-factor", type=int, default=2,
                        help="Super-resolution upscale factor")
    
    # Resolution
    parser.add_argument("--sensor-width", type=int, default=128,
                        help="LR sensor width in pixels")
    parser.add_argument("--sensor-height", type=int, default=96,
                        help="LR sensor height in pixels")
    
    # PSF settings
    parser.add_argument("--psf-sigma", type=float, default=0.8,
                        help="PSF sigma in HR pixels (translation mode)")
    
    # Noise settings
    parser.add_argument("--noise-std", type=float, default=0.01,
                        help="Noise standard deviation (translation mode)")
    
    # IBP settings
    parser.add_argument("--ibp-iterations", type=int, default=0,
                        help="Number of IBP iterations (0 to disable)")
    
    # Warp mode lens params
    parser.add_argument("--focal-length", type=float, default=35.0)
    parser.add_argument("--object-distance", type=float, default=600.0)
    parser.add_argument("--f-number", type=float, default=4.0)
    parser.add_argument("--shift-amplitude", type=float, default=0.05,
                        help="Optotune tilt amplitude in degrees (warp mode)")
    
    # Paths
    parser.add_argument("--scene", type=str, default="resources/media/barcode.svg")
    parser.add_argument("--output-dir", type=str, default="output/sr_demo")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SHIFT-AND-ADD SUPER RESOLUTION DEMO")
    print(f"Mode: {args.sr_mode.upper()}")
    print("=" * 70)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    scene_path = Path(args.scene)
    if not scene_path.exists():
        print(f"Error: Scene file not found: {scene_path}")
        return 1
    
    lr_resolution = (args.sensor_width, args.sensor_height)
    hr_resolution = (args.sensor_width * args.upscale_factor, args.sensor_height * args.upscale_factor)
    
    # =========================
    # STEP 1: Generate LR frames
    # =========================
    print("\n" + "=" * 70)
    print("STEP 1: FORWARD MODEL - Generate LR Frames")
    print("=" * 70)
    
    hr_truth = None
    
    if args.sr_mode == 'translation':
        # Load HR truth
        hr_truth = load_hr_truth(scene_path, hr_resolution)
        print(f"  Loaded HR truth: {hr_truth.shape[1]} x {hr_truth.shape[0]}")
        
        # Save HR truth
        cv2.imwrite(str(output_dir / "hr_truth.png"), (hr_truth * 255).astype(np.uint8))
        
        # Generate LR frames with known shifts
        frames, shifts, hr_blurred = generate_translation_frames(
            hr_truth,
            n_frames=args.n_frames,
            upscale_factor=args.upscale_factor,
            psf_sigma_hr_px=args.psf_sigma,
            noise_std=args.noise_std,
        )
        
        # Save LR frames
        for i, frame in enumerate(frames):
            cv2.imwrite(str(frames_dir / f"frame_{i:02d}.png"), (frame * 255).astype(np.uint8))
        
        warp_fields = None
        
    else:  # warp mode
        lens_params = {
            'focal_length_mm': args.focal_length,
            'object_distance_mm': args.object_distance,
            'f_number': args.f_number,
            'wavelength_nm': 550,
            'shift_amplitude_deg': args.shift_amplitude,
        }
        
        frames, warp_fields, metadata = generate_warp_frames(
            scene_path, frames_dir, lens_params,
            n_frames=args.n_frames,
            sensor_resolution=lr_resolution,
        )
        
        # Save LR frames
        for i, frame in enumerate(frames):
            cv2.imwrite(str(frames_dir / f"frame_{i:02d}.png"), (frame * 255).astype(np.uint8))
        
        shifts = None
    
    print(f"\n  Saved {len(frames)} LR frames to {frames_dir}")
    
    # =========================
    # STEP 2: SR Reconstruction
    # =========================
    print("\n" + "=" * 70)
    print("STEP 2: RECONSTRUCTION - Shift-and-Add")
    print("=" * 70)
    
    if args.sr_mode == 'translation':
        sr_image, weight_map = reconstruct_sr_translation(
            frames, shifts, args.upscale_factor
        )
    else:
        sr_image, weight_map = reconstruct_sr_warp(
            frames, warp_fields, args.upscale_factor
        )
    
    # Save SR result
    cv2.imwrite(str(output_dir / "sr_result.png"), (np.clip(sr_image, 0, 1) * 255).astype(np.uint8))
    
    # Save weight map
    weight_norm = weight_map / weight_map.max() if weight_map.max() > 0 else weight_map
    cv2.imwrite(str(output_dir / "weight_map.png"), (weight_norm * 255).astype(np.uint8))
    
    # =========================
    # STEP 3: IBP (optional)
    # =========================
    sr_ibp = None
    if args.ibp_iterations > 0 and args.sr_mode == 'translation':
        print("\n" + "=" * 70)
        print("STEP 3: ITERATIVE BACK PROJECTION")
        print("=" * 70)
        
        sr_ibp = ibp_translation(
            sr_image, frames, shifts, args.upscale_factor,
            psf_sigma_hr_px=args.psf_sigma,
            num_iterations=args.ibp_iterations,
        )
        
        cv2.imwrite(str(output_dir / "sr_ibp.png"), (np.clip(sr_ibp, 0, 1) * 255).astype(np.uint8))
    elif args.ibp_iterations > 0 and args.sr_mode == 'warp':
        print("\n  IBP disabled for warp mode (not implemented)")
    
    # =========================
    # STEP 4: Comparison
    # =========================
    print("\n" + "=" * 70)
    print("STEP 4: COMPARISON FIGURES")
    print("=" * 70)
    
    # Bicubic baseline
    bicubic = cv2.resize(frames[0], hr_resolution, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(output_dir / "bicubic.png"), (bicubic * 255).astype(np.uint8))
    
    # Compute metrics first (translation mode only)
    all_metrics = None
    if hr_truth is not None:
        try:
            all_metrics = {}
            all_metrics['bicubic'] = compute_metrics(np.clip(bicubic, 0, 1), hr_truth)
            all_metrics['sr'] = compute_metrics(np.clip(sr_image, 0, 1), hr_truth)
            if sr_ibp is not None:
                all_metrics['sr_ibp'] = compute_metrics(np.clip(sr_ibp, 0, 1), hr_truth)
            
            print(f"\n  Metrics vs HR truth:")
            print(f"    Bicubic:  PSNR={all_metrics['bicubic']['psnr']:.2f} dB, SSIM={all_metrics['bicubic']['ssim']:.4f}")
            print(f"    SR:       PSNR={all_metrics['sr']['psnr']:.2f} dB, SSIM={all_metrics['sr']['ssim']:.4f} (+{all_metrics['sr']['psnr'] - all_metrics['bicubic']['psnr']:.2f} dB)")
            if sr_ibp is not None:
                print(f"    SR+IBP:   PSNR={all_metrics['sr_ibp']['psnr']:.2f} dB, SSIM={all_metrics['sr_ibp']['ssim']:.4f} (+{all_metrics['sr_ibp']['psnr'] - all_metrics['bicubic']['psnr']:.2f} dB)")
        except ImportError:
            print("  (skimage not available for metrics)")
    
    # Comparison figure with metrics
    save_comparison_figure(
        frames[0], sr_image, bicubic, output_dir / "comparison.png",
        hr_truth=hr_truth, sr_ibp=sr_ibp, metrics=all_metrics,
    )
    
    save_zoomed_comparison(
        frames[0], sr_image, bicubic, output_dir / "comparison_zoom.png",
        hr_truth=hr_truth, sr_ibp=sr_ibp,
    )
    
    # =========================
    # Summary
    # =========================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Mode: {args.sr_mode}")
    print(f"  Input frames: {len(frames)}")
    print(f"  LR resolution: {lr_resolution[0]} x {lr_resolution[1]}")
    print(f"  HR resolution: {hr_resolution[0]} x {hr_resolution[1]}")
    print(f"  Upscale factor: {args.upscale_factor}x")
    if args.ibp_iterations > 0 and args.sr_mode == 'translation':
        print(f"  IBP iterations: {args.ibp_iterations}")
    print(f"\n  Outputs saved to: {output_dir}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
