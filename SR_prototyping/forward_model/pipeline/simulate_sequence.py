"""
Multi-frame sequence simulation for super-resolution.

Generates sequences of frames with different Optotune shift patterns
for super-resolution reconstruction.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union, Generator
import numpy as np

from .simulate_frame import FrameSimulator, DebugOutput
from ..optotune.plate_model import PlateModel


class ShiftPattern(Enum):
    """Predefined Optotune shift patterns."""
    
    # 2x2 pattern (4 frames)
    MANHATTAN_2X2 = "manhattan_2x2"
    DIAGONAL_2X2 = "diagonal_2x2"
    
    # 3x3 pattern (9 frames)
    GRID_3X3 = "grid_3x3"
    
    # 4x4 pattern (16 frames)
    GRID_4X4 = "grid_4x4"
    
    # Diamond/diagonal patterns
    DIAMOND_4 = "diamond_4"
    DIAMOND_8 = "diamond_8"
    
    # Custom (user-provided)
    CUSTOM = "custom"


@dataclass
class SequenceSimulator:
    """
    Simulates sequences of shifted frames for super-resolution.
    
    Attributes:
        frame_simulator: Base frame simulator
        shift_pattern: Pattern of Optotune tilts to apply
        custom_tilts: List of (tilt_x, tilt_y) for custom pattern
        shift_amplitude_deg: Maximum tilt amplitude for predefined patterns
    """
    frame_simulator: FrameSimulator
    shift_pattern: ShiftPattern = ShiftPattern.MANHATTAN_2X2
    custom_tilts: Optional[List[Tuple[float, float]]] = None
    shift_amplitude_deg: float = 0.5
    
    def __post_init__(self):
        self._tilts = self._generate_tilts()
    
    def _generate_tilts(self) -> List[Tuple[float, float]]:
        """Generate tilt sequence based on pattern."""
        amp = self.shift_amplitude_deg
        
        if self.shift_pattern == ShiftPattern.CUSTOM:
            if self.custom_tilts is None:
                raise ValueError("custom_tilts required for CUSTOM pattern")
            return self.custom_tilts
        
        elif self.shift_pattern == ShiftPattern.MANHATTAN_2X2:
            # 2x2 grid: (0,0), (1,0), (0,1), (1,1) normalized
            return [
                (0, 0),
                (amp, 0),
                (0, amp),
                (amp, amp),
            ]
        
        elif self.shift_pattern == ShiftPattern.DIAGONAL_2X2:
            # Diamond pattern for 2x2
            return [
                (0, 0),
                (amp, 0),
                (0, amp),
                (-amp, 0),
            ]
        
        elif self.shift_pattern == ShiftPattern.DIAMOND_4:
            # 4-point diamond
            return [
                (0, 0),
                (amp, 0),
                (0, amp),
                (-amp, 0),
            ]
        
        elif self.shift_pattern == ShiftPattern.DIAMOND_8:
            # 8-point diamond (includes diagonals)
            return [
                (0, 0),
                (amp, 0),
                (amp * 0.707, amp * 0.707),
                (0, amp),
                (-amp * 0.707, amp * 0.707),
                (-amp, 0),
                (-amp * 0.707, -amp * 0.707),
                (0, -amp),
            ]
        
        elif self.shift_pattern == ShiftPattern.GRID_3X3:
            # 3x3 grid
            tilts = []
            for j in range(3):
                for i in range(3):
                    tx = (i - 1) * amp
                    ty = (j - 1) * amp
                    tilts.append((tx, ty))
            return tilts
        
        elif self.shift_pattern == ShiftPattern.GRID_4X4:
            # 4x4 grid
            tilts = []
            for j in range(4):
                for i in range(4):
                    tx = (i - 1.5) * amp / 1.5
                    ty = (j - 1.5) * amp / 1.5
                    tilts.append((tx, ty))
            return tilts
        
        else:
            raise ValueError(f"Unknown shift pattern: {self.shift_pattern}")
    
    @property
    def num_frames(self) -> int:
        """Number of frames in the sequence."""
        return len(self._tilts)
    
    @property
    def tilts(self) -> List[Tuple[float, float]]:
        """List of (tilt_x, tilt_y) for each frame."""
        return self._tilts
    
    def simulate(
        self,
        scene: Union[np.ndarray, str, Path],
        debug: bool = False,
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[DebugOutput]]]:
        """
        Simulate the complete sequence.
        
        Args:
            scene: Input scene
            debug: Return debug outputs for each frame
        
        Returns:
            If debug=False: List of simulated frames
            If debug=True: (frames, debug_outputs)
        """
        frames = []
        debug_outputs = [] if debug else None
        
        for tilt_x, tilt_y in self._tilts:
            if debug:
                frame, dbg = self.frame_simulator.simulate_with_shift(
                    scene, tilt_x, tilt_y, debug=True
                )
                frames.append(frame)
                debug_outputs.append(dbg)
            else:
                frame = self.frame_simulator.simulate_with_shift(
                    scene, tilt_x, tilt_y, debug=False
                )
                frames.append(frame)
        
        if debug:
            return (frames, debug_outputs)
        return frames
    
    def simulate_generator(
        self,
        scene: Union[np.ndarray, str, Path],
    ) -> Generator[Tuple[int, np.ndarray, Tuple[float, float]], None, None]:
        """
        Generator version for memory-efficient processing.
        
        Yields:
            (frame_index, frame, (tilt_x, tilt_y))
        """
        for i, (tilt_x, tilt_y) in enumerate(self._tilts):
            frame = self.frame_simulator.simulate_with_shift(
                scene, tilt_x, tilt_y, debug=False
            )
            yield (i, frame, (tilt_x, tilt_y))
    
    def get_shift_vectors_px(self) -> List[Tuple[float, float]]:
        """
        Get the expected shift in sensor pixels for each frame.
        
        This is useful for registration in reconstruction algorithms.
        
        Returns:
            List of (dx_px, dy_px) expected shifts
        """
        shifts = []
        
        for tilt_x, tilt_y in self._tilts:
            if self.frame_simulator.optotune is None:
                # Create temporary plate for calculation
                plate = PlateModel(
                    thickness_mm=2.0,
                    refractive_index=1.5,
                    tilt_x_deg=tilt_x,
                    tilt_y_deg=tilt_y,
                )
            else:
                # Use configured plate with new tilts
                plate = PlateModel(
                    thickness_mm=self.frame_simulator.optotune.thickness_mm,
                    refractive_index=self.frame_simulator.optotune.refractive_index,
                    tilt_x_deg=tilt_x,
                    tilt_y_deg=tilt_y,
                    placement=self.frame_simulator.optotune.placement,
                )
            
            # Get on-axis shift
            dx_mm, dy_mm = plate.uniform_shift()
            
            # Convert to sensor pixels
            pixel_pitch_mm = self.frame_simulator.scene_units._sensor_pixel_pitch_mm
            dx_px = dx_mm / pixel_pitch_mm
            dy_px = dy_mm / pixel_pitch_mm
            
            shifts.append((dx_px, dy_px))
        
        return shifts


def simulate_sequence(
    scene: Union[np.ndarray, str, Path],
    pattern: Union[ShiftPattern, str] = ShiftPattern.MANHATTAN_2X2,
    shift_amplitude_deg: float = 0.5,
    custom_tilts: Optional[List[Tuple[float, float]]] = None,
    **frame_kwargs,
) -> List[np.ndarray]:
    """
    Convenience function to simulate a shifted frame sequence.
    
    Args:
        scene: Input scene
        pattern: Shift pattern (ShiftPattern enum or string name)
        shift_amplitude_deg: Maximum tilt for predefined patterns
        custom_tilts: Custom tilt list for CUSTOM pattern
        **frame_kwargs: Arguments passed to FrameSimulator
    
    Returns:
        List of simulated frames
    """
    from .simulate_frame import simulate_frame
    
    # Handle string pattern names
    if isinstance(pattern, str):
        pattern = ShiftPattern(pattern)
    
    # Build a frame simulator
    # First, simulate one frame to get the simulator configured
    from ..scene.scene_units import SceneUnits, PhysicalExtent
    from ..optics.lens_model import ThinLensModel
    
    # Extract parameters with defaults
    sensor_resolution = frame_kwargs.get('sensor_resolution', (2048, 1536))
    sensor_pixel_pitch_um = frame_kwargs.get('sensor_pixel_pitch_um', 3.45)
    oversampling_factor = frame_kwargs.get('oversampling_factor', 4)
    focal_length_mm = frame_kwargs.get('focal_length_mm', 25.0)
    object_distance_mm = frame_kwargs.get('object_distance_mm', 100.0)
    f_number = frame_kwargs.get('f_number', 2.8)
    scene_width_mm = frame_kwargs.get('scene_width_mm', None)
    exposure_time_s = frame_kwargs.get('exposure_time_s', 0.001)
    seed = frame_kwargs.get('seed', None)
    
    # Create lens model
    lens = ThinLensModel(
        focal_length_mm=focal_length_mm,
        object_distance_mm=object_distance_mm,
        f_number=f_number,
    )
    
    # Compute scene dimensions
    if scene_width_mm is None:
        sensor_width_mm = sensor_resolution[0] * sensor_pixel_pitch_um / 1000
        scene_width_mm = sensor_width_mm / lens.magnification()
    
    aspect = sensor_resolution[1] / sensor_resolution[0]
    scene_height_mm = scene_width_mm * aspect
    
    # Create scene units
    scene_units = SceneUnits(
        physical_extent=PhysicalExtent(
            width_mm=scene_width_mm,
            height_mm=scene_height_mm,
        ),
        oversampling_factor=oversampling_factor,
        sensor_pixel_pitch_um=sensor_pixel_pitch_um,
        magnification=lens.magnification(),
    )
    
    # Create base plate model
    optotune = PlateModel(
        thickness_mm=2.0,
        refractive_index=1.5,
        tilt_x_deg=0,
        tilt_y_deg=0,
    )
    
    # Create frame simulator
    frame_sim = FrameSimulator(
        scene_units=scene_units,
        lens=lens,
        optotune=optotune,
        exposure_time_s=exposure_time_s,
        random_seed=seed,
    )
    
    # Create sequence simulator
    seq_sim = SequenceSimulator(
        frame_simulator=frame_sim,
        shift_pattern=pattern,
        custom_tilts=custom_tilts,
        shift_amplitude_deg=shift_amplitude_deg,
    )
    
    return seq_sim.simulate(scene, debug=False)
