"""
Single frame simulation through the complete optical chain.

Orchestrates the full pipeline:
Scene -> PSF Blur -> Motion Blur -> Optotune Warp -> Pixel Integration -> Noise -> ADC
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
import numpy as np

from ..scene.scene_units import SceneUnits, PhysicalExtent
from ..scene.svg_loader import SVGLoader, load_svg
from ..optics.lens_model import ThinLensModel
from ..optics.psf_model import PSFModel, DiffractionPSF, GaussianPSF
from ..optics.motion_blur import MotionBlurModel
from ..optotune.plate_model import PlateModel
from ..optotune.warp_field import WarpField, create_warp_field, create_uniform_warp
from ..sensor.pixel_integration import PixelIntegrator
from ..sensor.noise_model import NoiseModel, SensorNoise
from ..sensor.adc_model import ADCModel


@dataclass
class DebugOutput:
    """
    Container for intermediate pipeline outputs.
    
    Useful for visualization and debugging each stage of the simulation.
    """
    scene_raster: Optional[np.ndarray] = None
    after_psf: Optional[np.ndarray] = None
    after_motion: Optional[np.ndarray] = None
    after_warp: Optional[np.ndarray] = None
    after_integration: Optional[np.ndarray] = None
    after_noise: Optional[np.ndarray] = None
    final_dn: Optional[np.ndarray] = None
    warp_field: Optional[WarpField] = None
    
    def save_all(self, output_dir: Union[str, Path], prefix: str = "debug") -> None:
        """Save all intermediate outputs as images."""
        import cv2
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        def save_normalized(arr: Optional[np.ndarray], name: str) -> None:
            if arr is None:
                return
            # Normalize to 0-255
            arr_norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-10) * 255)
            cv2.imwrite(str(output_dir / f"{prefix}_{name}.png"), arr_norm.astype(np.uint8))
        
        save_normalized(self.scene_raster, "01_scene")
        save_normalized(self.after_psf, "02_psf")
        save_normalized(self.after_motion, "03_motion")
        save_normalized(self.after_warp, "04_warp")
        save_normalized(self.after_integration, "05_integration")
        save_normalized(self.after_noise, "06_noise")
        
        if self.final_dn is not None:
            cv2.imwrite(
                str(output_dir / f"{prefix}_07_final.png"),
                (self.final_dn / self.final_dn.max() * 255).astype(np.uint8)
            )
        
        if self.warp_field is not None:
            flow_vis = self.warp_field.to_flow_visualization()
            cv2.imwrite(str(output_dir / f"{prefix}_warp_field.png"), flow_vis[..., ::-1])


@dataclass
class FrameSimulator:
    """
    Complete frame simulation pipeline.
    
    Configures all components and provides methods to simulate frames
    with different inputs and settings.
    
    Attributes:
        scene_units: Coordinate system and resolution settings
        lens: Lens model for geometry and PSF
        psf: PSF model (auto-generated from lens if None)
        optotune: Optotune plate model (None for no shift)
        noise: Sensor noise parameters
        adc: ADC model
        motion: Motion blur model (None for static scene)
    """
    # Coordinate system
    scene_units: SceneUnits
    
    # Optics
    lens: ThinLensModel
    psf: Optional[PSFModel] = None
    
    # Optotune (optional)
    optotune: Optional[PlateModel] = None
    
    # Sensor
    noise: Optional[SensorNoise] = None
    adc: Optional[ADCModel] = None
    exposure_time_s: float = 0.001
    
    # Motion (optional)
    motion: Optional[MotionBlurModel] = None
    
    # Processing options
    photons_at_saturation: float = 10000
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        # Auto-generate PSF from lens if not provided
        if self.psf is None:
            self.psf = DiffractionPSF(
                wavelength_nm=self.lens.wavelength_nm,
                f_number=self.lens.f_number,
            )
        
        # Default noise if not provided
        if self.noise is None:
            self.noise = SensorNoise()
        
        # Default ADC if not provided
        if self.adc is None:
            self.adc = ADCModel()
        
        # Create pixel integrator
        self._integrator = PixelIntegrator(
            oversampling_factor=self.scene_units.oversampling_factor
        )
        
        # Create noise model
        self._noise_model = NoiseModel(
            noise_params=self.noise,
            exposure_time_s=self.exposure_time_s,
            seed=self.random_seed,
        )
    
    def simulate(
        self,
        scene: Union[np.ndarray, str, Path],
        debug: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, DebugOutput]]:
        """
        Simulate a single frame through the complete pipeline.
        
        Args:
            scene: Input scene as:
                - numpy array (already at internal resolution)
                - path to SVG file
            debug: If True, return intermediate outputs
        
        Returns:
            If debug=False: Final digital output (uint16)
            If debug=True: (final_output, DebugOutput)
        """
        debug_out = DebugOutput() if debug else None
        
        # 1. Load/prepare scene
        if isinstance(scene, (str, Path)):
            scene_img = load_svg(scene, self.scene_units)
        else:
            scene_img = scene.astype(np.float32)
        
        if debug:
            debug_out.scene_raster = scene_img.copy()
        
        # 2. Apply PSF blur
        pixel_pitch_um = self.scene_units.internal_pixel_pitch_mm * 1000
        after_psf = self.psf.apply(scene_img, pixel_pitch_um)
        
        if debug:
            debug_out.after_psf = after_psf.copy()
        
        # 3. Apply motion blur (if configured)
        if self.motion is not None:
            after_motion = self.motion.apply(after_psf, pixel_pitch_um)
        else:
            after_motion = after_psf
        
        if debug:
            debug_out.after_motion = after_motion.copy()
        
        # 4. Apply Optotune warp (if configured)
        if self.optotune is not None:
            warp = create_warp_field(
                self.optotune,
                self.scene_units,
                self.lens.image_distance_mm(),
            )
            after_warp = warp.apply(after_motion)
            
            if debug:
                debug_out.warp_field = warp
        else:
            after_warp = after_motion
        
        if debug:
            debug_out.after_warp = after_warp.copy()
        
        # 5. Pixel integration (downsample to sensor resolution)
        after_integration = self._integrator.integrate(after_warp)
        
        if debug:
            debug_out.after_integration = after_integration.copy()
        
        # 6. Convert to photons and apply noise
        photons = after_integration * self.photons_at_saturation
        electrons = self._noise_model.apply(photons)
        
        if debug:
            debug_out.after_noise = electrons.copy()
        
        # 7. Apply ADC
        final_dn = self.adc.apply(electrons)
        
        if debug:
            debug_out.final_dn = final_dn.copy()
            return (final_dn, debug_out)
        
        return final_dn
    
    def simulate_with_shift(
        self,
        scene: Union[np.ndarray, str, Path],
        tilt_x_deg: float = 0.0,
        tilt_y_deg: float = 0.0,
        debug: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, DebugOutput]]:
        """
        Simulate a frame with specific Optotune tilt angles.
        
        Convenience method that temporarily overrides the Optotune settings.
        
        Args:
            scene: Input scene
            tilt_x_deg: X-axis tilt in degrees
            tilt_y_deg: Y-axis tilt in degrees
            debug: Return debug outputs
        
        Returns:
            Simulated frame
        """
        # Create temporary plate model with specified tilts
        if self.optotune is not None:
            original_x = self.optotune.tilt_x_deg
            original_y = self.optotune.tilt_y_deg
            self.optotune.tilt_x_deg = tilt_x_deg
            self.optotune.tilt_y_deg = tilt_y_deg
        else:
            # Create default plate model
            self.optotune = PlateModel(
                thickness_mm=2.0,
                refractive_index=1.5,
                tilt_x_deg=tilt_x_deg,
                tilt_y_deg=tilt_y_deg,
            )
            original_x = original_y = None
        
        try:
            result = self.simulate(scene, debug=debug)
        finally:
            # Restore original settings
            if original_x is not None:
                self.optotune.tilt_x_deg = original_x
                self.optotune.tilt_y_deg = original_y
        
        return result


def simulate_frame(
    scene: Union[np.ndarray, str, Path],
    sensor_resolution: Tuple[int, int] = (2048, 1536),
    sensor_pixel_pitch_um: float = 3.45,
    oversampling_factor: int = 4,
    focal_length_mm: float = 25.0,
    object_distance_mm: float = 100.0,
    f_number: float = 2.8,
    scene_width_mm: Optional[float] = None,
    optotune_tilt_x_deg: float = 0.0,
    optotune_tilt_y_deg: float = 0.0,
    exposure_time_s: float = 0.001,
    velocity_mm_s: Tuple[float, float] = (0.0, 0.0),
    debug: bool = False,
    seed: Optional[int] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, DebugOutput]]:
    """
    Convenience function to simulate a single frame with default parameters.
    
    Args:
        scene: Input scene (numpy array or SVG path)
        sensor_resolution: (width, height) in pixels
        sensor_pixel_pitch_um: Sensor pixel size
        oversampling_factor: Internal oversampling for subpixel accuracy
        focal_length_mm: Lens focal length
        object_distance_mm: Distance from object to lens
        f_number: Lens f-number
        scene_width_mm: Physical width of scene (auto if None)
        optotune_tilt_x_deg: Optotune X tilt
        optotune_tilt_y_deg: Optotune Y tilt
        exposure_time_s: Exposure time
        velocity_mm_s: Sample velocity (vx, vy)
        debug: Return debug outputs
        seed: Random seed
    
    Returns:
        Simulated frame (and debug outputs if requested)
    """
    # Create lens model
    lens = ThinLensModel(
        focal_length_mm=focal_length_mm,
        object_distance_mm=object_distance_mm,
        f_number=f_number,
    )
    
    # Compute scene dimensions
    if scene_width_mm is None:
        # Compute from sensor and magnification
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
    
    # Create Optotune model if tilts are specified
    optotune = None
    if optotune_tilt_x_deg != 0 or optotune_tilt_y_deg != 0:
        optotune = PlateModel(
            thickness_mm=2.0,
            refractive_index=1.5,
            tilt_x_deg=optotune_tilt_x_deg,
            tilt_y_deg=optotune_tilt_y_deg,
        )
    
    # Create motion model if velocity is specified
    motion = None
    if velocity_mm_s[0] != 0 or velocity_mm_s[1] != 0:
        motion = MotionBlurModel(
            velocity_mm_s=velocity_mm_s,
            exposure_time_s=exposure_time_s,
            magnification=lens.magnification(),
        )
    
    # Create simulator
    simulator = FrameSimulator(
        scene_units=scene_units,
        lens=lens,
        optotune=optotune,
        motion=motion,
        exposure_time_s=exposure_time_s,
        random_seed=seed,
    )
    
    return simulator.simulate(scene, debug=debug)
