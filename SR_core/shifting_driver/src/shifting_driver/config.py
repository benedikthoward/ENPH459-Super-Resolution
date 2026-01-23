"""
Configuration loader for the shifting driver.

Loads settings from the shared config.toml file in SR_core/runner/.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def find_config_file() -> Path:
    """
    Find the config.toml file in SR_core/runner/.
    
    Returns:
        Path to config.toml
        
    Raises:
        FileNotFoundError: If config.toml cannot be found.
    """
    # Start from this file's directory (shifting_driver package)
    # Path: SR_core/shifting_driver/src/shifting_driver/config.py
    # Config: SR_core/runner/config.toml
    this_file = Path(__file__).resolve()
    
    # Navigate up to SR_core, then into runner
    sr_core_dir = this_file.parent.parent.parent.parent  # Up 4 levels to SR_core
    config_path = sr_core_dir / "runner" / "config.toml"
    
    if config_path.exists():
        return config_path
    
    # Fallback: search upward for SR_core/runner/config.toml
    current = this_file.parent
    for _ in range(10):
        candidate = current / "runner" / "config.toml"
        if candidate.exists():
            return candidate
        
        # Also check if we're in SR_core
        if current.name == "SR_core":
            candidate = current / "runner" / "config.toml"
            if candidate.exists():
                return candidate
        
        parent = current.parent
        if parent == current:
            break
        current = parent
    
    raise FileNotFoundError(
        "Could not find config.toml. "
        "Ensure it exists at SR_core/runner/config.toml"
    )


@dataclass
class IPCConfig:
    """IPC configuration."""
    socket_path: str
    timeout_seconds: float


@dataclass
class CameraConfig:
    """Camera configuration."""
    device: str  # Device index ("0", "1") or serial number
    resolution: tuple[int, int]
    frame_rate: int
    pixel_format: str
    exposure_us: int
    gain_db: float


@dataclass
class BeamShifterConfig:
    """Beam shifter configuration."""
    default_frame_rate: int
    default_waveform: str
    channel: int
    port: Optional[str]


@dataclass
class StageConfig:
    """Zaber stage configuration."""
    port: Optional[str]
    axis: int


@dataclass
class OutputConfig:
    """Output configuration."""
    image_dir: str
    log_level: str


@dataclass
class Config:
    """Complete configuration."""
    ipc: IPCConfig
    camera: CameraConfig
    beam_shifter: BeamShifterConfig
    stage: StageConfig
    output: OutputConfig
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """
        Load configuration from TOML file.
        
        Args:
            config_path: Path to config file. If None, searches for config.toml.
            
        Returns:
            Loaded Config instance.
        """
        if config_path is None:
            config_path = find_config_file()
        
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        
        ipc_data = data.get("ipc", {})
        camera_data = data.get("camera", {})
        shifter_data = data.get("beam_shifter", {})
        stage_data = data.get("stage", {})
        output_data = data.get("output", {})
        
        return cls(
            ipc=IPCConfig(
                socket_path=ipc_data.get("socket_path", "/tmp/sr_ipc.sock"),
                timeout_seconds=ipc_data.get("timeout_seconds", 5.0),
            ),
            camera=CameraConfig(
                device=str(camera_data.get("device", "0")),
                resolution=tuple(camera_data.get("resolution", [2048, 1536])),
                frame_rate=camera_data.get("frame_rate", 56),
                pixel_format=camera_data.get("pixel_format", "Mono8"),
                exposure_us=camera_data.get("exposure_us", 10000),
                gain_db=camera_data.get("gain_db", 0.0),
            ),
            beam_shifter=BeamShifterConfig(
                default_frame_rate=shifter_data.get("default_frame_rate", 60),
                default_waveform=shifter_data.get("default_waveform", "manhattan"),
                channel=shifter_data.get("channel", 0),
                port=shifter_data.get("port") or None,
            ),
            stage=StageConfig(
                port=stage_data.get("port") or None,
                axis=stage_data.get("axis", 1),
            ),
            output=OutputConfig(
                image_dir=output_data.get("image_dir", "./output/frames"),
                log_level=output_data.get("log_level", "info"),
            ),
        )


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """
    Get the global configuration instance.
    
    Loads config on first call, returns cached instance on subsequent calls.
    
    Args:
        config_path: Optional path to config file (only used on first call).
        
    Returns:
        Config instance.
    """
    global _config
    if _config is None:
        _config = Config.load(config_path)
    return _config


def reload_config(config_path: Optional[Path] = None) -> Config:
    """
    Force reload of configuration.
    
    Args:
        config_path: Optional path to config file.
        
    Returns:
        Newly loaded Config instance.
    """
    global _config
    _config = Config.load(config_path)
    return _config
