"""
Configuration loading and management.

Provides utilities to load YAML config files and construct simulation objects.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml

from ..scene.scene_units import SceneUnits, PhysicalExtent
from ..optics.lens_model import ThinLensModel
from ..optics.psf_model import DiffractionPSF, GaussianPSF, ZernikePSF
from ..optotune.plate_model import PlateModel
from ..sensor.noise_model import SensorNoise
from ..sensor.adc_model import ADCModel
from ..pipeline.simulate_frame import FrameSimulator


# Path to configs directory
CONFIGS_DIR = Path(__file__).parent


def get_config_path(category: str, name: str) -> Path:
    """
    Get path to a config file.
    
    Args:
        category: Config category ('sensors', 'lenses', 'optotune', 'stage')
        name: Config name (without .yaml extension)
    
    Returns:
        Path to config file
    """
    path = CONFIGS_DIR / category / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return path


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_sensor_config(name: str = "imx265") -> Dict[str, Any]:
    """Load a sensor configuration."""
    return load_config(get_config_path("sensors", name))


def load_lens_config(name: str = "default") -> Dict[str, Any]:
    """Load a lens configuration."""
    return load_config(get_config_path("lenses", name))


def load_optotune_config(name: str = "xrp20") -> Dict[str, Any]:
    """Load an Optotune configuration."""
    return load_config(get_config_path("optotune", name))


def load_stage_config(name: str = "zaber") -> Dict[str, Any]:
    """Load a stage configuration."""
    return load_config(get_config_path("stage", name))


def create_lens_from_config(
    config: Optional[Dict[str, Any]] = None,
    object_distance_mm: Optional[float] = None,
) -> ThinLensModel:
    """
    Create a ThinLensModel from config.
    
    Args:
        config: Lens config dict. If None, loads default.
        object_distance_mm: Override object distance.
    
    Returns:
        Configured ThinLensModel
    """
    if config is None:
        config = load_lens_config()
    
    lens_cfg = config.get('lens', config)
    obj_cfg = config.get('object_space', {})
    
    return ThinLensModel(
        focal_length_mm=lens_cfg.get('focal_length_mm', 25.0),
        object_distance_mm=object_distance_mm or obj_cfg.get('working_distance_mm', 100.0),
        f_number=lens_cfg.get('working_f_number', lens_cfg.get('f_number', 2.8)),
        wavelength_nm=lens_cfg.get('wavelength_nm', 550),
    )


def create_psf_from_config(
    config: Optional[Dict[str, Any]] = None,
    lens: Optional[ThinLensModel] = None,
):
    """
    Create a PSF model from config.
    
    Args:
        config: Lens/PSF config dict. If None, loads default.
        lens: Lens model to get parameters from.
    
    Returns:
        Configured PSF model
    """
    if config is None:
        config = load_lens_config()
    
    psf_cfg = config.get('psf', {})
    mode = psf_cfg.get('mode', 'diffraction')
    
    # Get wavelength and f_number from lens or config
    if lens is not None:
        wavelength_nm = lens.wavelength_nm
        f_number = lens.f_number
    else:
        lens_cfg = config.get('lens', {})
        wavelength_nm = lens_cfg.get('wavelength_nm', 550)
        f_number = lens_cfg.get('f_number', 2.8)
    
    if mode == 'diffraction':
        return DiffractionPSF(
            wavelength_nm=wavelength_nm,
            f_number=f_number,
            defocus_waves=psf_cfg.get('defocus_waves', 0.0),
        )
    elif mode == 'gaussian':
        return GaussianPSF.from_lens(wavelength_nm, f_number)
    elif mode == 'zernike':
        zernike_cfg = config.get('zernike_coeffs', {})
        return ZernikePSF(
            wavelength_nm=wavelength_nm,
            f_number=f_number,
            zernike_coeffs={int(k): v for k, v in zernike_cfg.items()},
        )
    else:
        raise ValueError(f"Unknown PSF mode: {mode}")


def create_sensor_noise_from_config(
    config: Optional[Dict[str, Any]] = None,
) -> SensorNoise:
    """Create SensorNoise from config."""
    if config is None:
        config = load_sensor_config()
    
    noise_cfg = config.get('noise', config.get('sensor', {}))
    
    return SensorNoise(
        read_noise_e=noise_cfg.get('read_noise_e', 2.5),
        dark_current_e_per_s=noise_cfg.get('dark_current_e_per_s', 0.5),
        fixed_pattern_noise_percent=noise_cfg.get('fixed_pattern_noise_percent', 0.0),
        quantum_efficiency=noise_cfg.get('quantum_efficiency', noise_cfg.get('qe', 0.7)),
    )


def create_adc_from_config(
    config: Optional[Dict[str, Any]] = None,
) -> ADCModel:
    """Create ADCModel from config."""
    if config is None:
        config = load_sensor_config()
    
    adc_cfg = config.get('adc', {})
    sensor_cfg = config.get('sensor', {})
    
    return ADCModel(
        bit_depth=adc_cfg.get('bit_depth', sensor_cfg.get('bit_depth', 12)),
        gain=adc_cfg.get('gain', 1.0),
        black_level_dn=adc_cfg.get('black_level_dn', 64),
        full_well_e=sensor_cfg.get('full_well_e', 10000),
    )


def create_optotune_from_config(
    config: Optional[Dict[str, Any]] = None,
    tilt_x_deg: float = 0.0,
    tilt_y_deg: float = 0.0,
) -> PlateModel:
    """Create PlateModel from config."""
    if config is None:
        config = load_optotune_config()
    
    opt_cfg = config.get('optotune', config)
    
    return PlateModel(
        thickness_mm=opt_cfg.get('plate_thickness_mm', 2.0),
        refractive_index=opt_cfg.get('refractive_index', 1.517),
        tilt_x_deg=tilt_x_deg,
        tilt_y_deg=tilt_y_deg,
        placement=opt_cfg.get('placement', 'converging'),
    )


def create_scene_units(
    sensor_config: Optional[Dict[str, Any]] = None,
    lens: Optional[ThinLensModel] = None,
    scene_width_mm: Optional[float] = None,
    oversampling_factor: int = 4,
) -> SceneUnits:
    """
    Create SceneUnits from configs.
    
    Args:
        sensor_config: Sensor config dict
        lens: Lens model (for magnification)
        scene_width_mm: Physical scene width (auto-computed if None)
        oversampling_factor: Internal oversampling
    
    Returns:
        Configured SceneUnits
    """
    if sensor_config is None:
        sensor_config = load_sensor_config()
    
    sensor = sensor_config.get('sensor', sensor_config)
    resolution = sensor.get('resolution', [2048, 1536])
    pixel_pitch_um = sensor.get('pixel_pitch_um', 3.45)
    
    if lens is None:
        lens = create_lens_from_config()
    
    mag = lens.magnification()
    
    # Compute scene size from sensor and magnification
    if scene_width_mm is None:
        sensor_width_mm = resolution[0] * pixel_pitch_um / 1000
        scene_width_mm = sensor_width_mm / mag
    
    aspect = resolution[1] / resolution[0]
    scene_height_mm = scene_width_mm * aspect
    
    return SceneUnits(
        physical_extent=PhysicalExtent(
            width_mm=scene_width_mm,
            height_mm=scene_height_mm,
        ),
        oversampling_factor=oversampling_factor,
        sensor_pixel_pitch_um=pixel_pitch_um,
        magnification=mag,
    )


def create_simulator_from_configs(
    sensor_name: str = "imx265",
    lens_name: str = "default",
    optotune_name: str = "xrp20",
    oversampling_factor: int = 4,
    exposure_time_s: float = 0.001,
    random_seed: Optional[int] = None,
) -> FrameSimulator:
    """
    Create a fully configured FrameSimulator from config files.
    
    Args:
        sensor_name: Sensor config name
        lens_name: Lens config name
        optotune_name: Optotune config name
        oversampling_factor: Internal oversampling
        exposure_time_s: Exposure time
        random_seed: Random seed for reproducibility
    
    Returns:
        Configured FrameSimulator
    """
    sensor_cfg = load_sensor_config(sensor_name)
    lens_cfg = load_lens_config(lens_name)
    optotune_cfg = load_optotune_config(optotune_name)
    
    lens = create_lens_from_config(lens_cfg)
    psf = create_psf_from_config(lens_cfg, lens)
    scene_units = create_scene_units(sensor_cfg, lens, oversampling_factor=oversampling_factor)
    noise = create_sensor_noise_from_config(sensor_cfg)
    adc = create_adc_from_config(sensor_cfg)
    optotune = create_optotune_from_config(optotune_cfg)
    
    return FrameSimulator(
        scene_units=scene_units,
        lens=lens,
        psf=psf,
        optotune=optotune,
        noise=noise,
        adc=adc,
        exposure_time_s=exposure_time_s,
        random_seed=random_seed,
    )
