#!/usr/bin/env python3
"""
Forward Model CLI Entry Point

Run forward model simulations from the command line with various options.

Usage:
    python -m scripts.run_forward_model --scene resources/media/barcode.svg --debug
    python -m scripts.run_forward_model --scene resources/media/barcode.svg --sequence --pattern manhattan_2x2
"""

import argparse
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Run forward model simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Path to SVG scene file or 'synthetic' for test pattern",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="sim",
        help="Prefix for output files",
    )
    
    # Mode
    parser.add_argument(
        "--sequence",
        action="store_true",
        help="Simulate a shifted frame sequence",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="manhattan_2x2",
        choices=["manhattan_2x2", "diamond_4", "grid_3x3", "grid_4x4", "diamond_8"],
        help="Shift pattern for sequence mode",
    )
    parser.add_argument(
        "--shift-amplitude",
        type=float,
        default=0.5,
        help="Shift amplitude in degrees",
    )
    
    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate outputs for debugging",
    )
    
    # Physics parameters
    parser.add_argument(
        "--focal-length",
        type=float,
        default=25.0,
        help="Lens focal length (mm)",
    )
    parser.add_argument(
        "--object-distance",
        type=float,
        default=100.0,
        help="Object distance (mm)",
    )
    parser.add_argument(
        "--f-number",
        type=float,
        default=2.8,
        help="Lens f-number",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=0.001,
        help="Exposure time (seconds)",
    )
    parser.add_argument(
        "--velocity-x",
        type=float,
        default=0.0,
        help="Sample X velocity (mm/s)",
    )
    parser.add_argument(
        "--velocity-y",
        type=float,
        default=0.0,
        help="Sample Y velocity (mm/s)",
    )
    
    # Resolution
    parser.add_argument(
        "--oversampling",
        type=int,
        default=4,
        help="Internal oversampling factor",
    )
    parser.add_argument(
        "--scene-width",
        type=float,
        default=None,
        help="Physical scene width (mm), auto if not specified",
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Import forward model (delayed to show help faster)
    from forward_model import simulate_frame, simulate_sequence
    from forward_model.pipeline import FrameSimulator, SequenceSimulator, ShiftPattern
    from forward_model.configs import create_simulator_from_configs
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine scene input
    if args.scene == "synthetic":
        # Generate synthetic test pattern
        scene = _create_test_pattern(1024, 768)
        print("Using synthetic test pattern")
    else:
        scene = Path(args.scene)
        if not scene.exists():
            print(f"Error: Scene file not found: {scene}")
            sys.exit(1)
        print(f"Loading scene: {scene}")
    
    # Build simulation parameters
    sim_kwargs = {
        "focal_length_mm": args.focal_length,
        "object_distance_mm": args.object_distance,
        "f_number": args.f_number,
        "exposure_time_s": args.exposure,
        "velocity_mm_s": (args.velocity_x, args.velocity_y),
        "oversampling_factor": args.oversampling,
        "scene_width_mm": args.scene_width,
        "seed": args.seed,
    }
    
    if args.sequence:
        # Simulate shifted sequence
        print(f"Simulating sequence with pattern: {args.pattern}")
        pattern = ShiftPattern(args.pattern)
        
        frames = simulate_sequence(
            scene,
            pattern=pattern,
            shift_amplitude_deg=args.shift_amplitude,
            **sim_kwargs,
        )
        
        # Save frames
        import cv2
        for i, frame in enumerate(frames):
            out_path = output_dir / f"{args.prefix}_frame_{i:02d}.png"
            # Normalize to 8-bit for saving
            frame_8bit = (frame / frame.max() * 255).astype(np.uint8)
            cv2.imwrite(str(out_path), frame_8bit)
            print(f"  Saved: {out_path}")
        
        print(f"Saved {len(frames)} frames to {output_dir}")
        
    else:
        # Single frame simulation
        print("Simulating single frame...")
        
        if args.debug:
            result, debug_out = simulate_frame(scene, debug=True, **sim_kwargs)
            
            # Save debug outputs
            debug_dir = output_dir / "debug"
            debug_out.save_all(debug_dir, args.prefix)
            print(f"Saved debug outputs to {debug_dir}")
        else:
            result = simulate_frame(scene, debug=False, **sim_kwargs)
        
        # Save final result
        import cv2
        out_path = output_dir / f"{args.prefix}_final.png"
        result_8bit = (result / result.max() * 255).astype(np.uint8)
        cv2.imwrite(str(out_path), result_8bit)
        print(f"Saved result to {out_path}")
    
    print("Done!")


def _create_test_pattern(width: int, height: int) -> np.ndarray:
    """Create a synthetic test pattern for simulation."""
    # Create resolution target pattern
    img = np.ones((height, width), dtype=np.float32) * 0.9
    
    # Add some bar patterns at different frequencies
    for freq in [8, 16, 32, 64]:
        y_start = int(height * (freq - 8) / 64)
        y_end = int(height * freq / 64)
        
        x = np.arange(width)
        pattern = (np.sin(2 * np.pi * freq * x / width) > 0).astype(np.float32)
        
        for y in range(y_start, min(y_end, height)):
            img[y, :] = pattern * 0.8 + 0.1
    
    # Add some point targets
    for i in range(5):
        cx = int(width * (i + 1) / 6)
        cy = int(height * 0.8)
        img[cy-2:cy+3, cx-2:cx+3] = 0.0
    
    return img


if __name__ == "__main__":
    main()
