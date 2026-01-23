//! Configuration loader for the image processing server.
//!
//! Loads settings from the shared config.toml file in SR_core/runner/.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tracing::info;

/// IPC configuration
#[derive(Debug, Deserialize, Clone)]
pub struct IpcConfig {
    /// Unix domain socket path
    #[serde(default = "default_socket_path")]
    pub socket_path: String,
    
    /// Connection timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_seconds: f64,
}

fn default_socket_path() -> String {
    "/tmp/sr_ipc.sock".to_string()
}

fn default_timeout() -> f64 {
    5.0
}

impl Default for IpcConfig {
    fn default() -> Self {
        Self {
            socket_path: default_socket_path(),
            timeout_seconds: default_timeout(),
        }
    }
}

/// Camera configuration
#[derive(Debug, Deserialize, Clone)]
pub struct CameraConfig {
    /// Device identifier: index ("0", "1") or serial number
    #[serde(default = "default_device")]
    pub device: String,
    
    /// Resolution [width, height]
    #[serde(default = "default_resolution")]
    pub resolution: [u32; 2],
    
    /// Frame rate in fps
    #[serde(default = "default_camera_frame_rate")]
    pub frame_rate: u32,
    
    /// Pixel format (e.g., "Mono8", "BayerRG8")
    #[serde(default = "default_pixel_format")]
    pub pixel_format: String,
    
    /// Exposure time in microseconds
    #[serde(default = "default_exposure")]
    pub exposure_us: u32,
    
    /// Gain in dB
    #[serde(default)]
    pub gain_db: f64,
}

fn default_device() -> String {
    "0".to_string()
}

fn default_resolution() -> [u32; 2] {
    [2048, 1536]
}

fn default_camera_frame_rate() -> u32 {
    56
}

fn default_pixel_format() -> String {
    "Mono8".to_string()
}

fn default_exposure() -> u32 {
    10000
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            device: default_device(),
            resolution: default_resolution(),
            frame_rate: default_camera_frame_rate(),
            pixel_format: default_pixel_format(),
            exposure_us: default_exposure(),
            gain_db: 0.0,
        }
    }
}

/// Beam shifter configuration
#[derive(Debug, Deserialize, Clone)]
pub struct BeamShifterConfig {
    /// Default frame rate in Hz
    #[serde(default = "default_shifter_frame_rate")]
    pub default_frame_rate: u32,
    
    /// Default waveform pattern
    #[serde(default = "default_waveform")]
    pub default_waveform: String,
    
    /// ICC-4C channel
    #[serde(default)]
    pub channel: u32,
    
    /// Serial port (empty for auto-detection)
    #[serde(default)]
    pub port: String,
}

fn default_shifter_frame_rate() -> u32 {
    60
}

fn default_waveform() -> String {
    "manhattan".to_string()
}

impl Default for BeamShifterConfig {
    fn default() -> Self {
        Self {
            default_frame_rate: default_shifter_frame_rate(),
            default_waveform: default_waveform(),
            channel: 0,
            port: String::new(),
        }
    }
}

/// Zaber stage configuration
#[derive(Debug, Deserialize, Clone)]
pub struct StageConfig {
    /// Serial port (empty for none)
    #[serde(default)]
    pub port: String,
    
    /// Axis number
    #[serde(default = "default_axis")]
    pub axis: u32,
}

fn default_axis() -> u32 {
    1
}

impl Default for StageConfig {
    fn default() -> Self {
        Self {
            port: String::new(),
            axis: default_axis(),
        }
    }
}

/// Output configuration
#[derive(Debug, Deserialize, Clone)]
pub struct OutputConfig {
    /// Directory for captured frames
    #[serde(default = "default_image_dir")]
    pub image_dir: String,
    
    /// Log level
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

fn default_image_dir() -> String {
    "./output/frames".to_string()
}

fn default_log_level() -> String {
    "info".to_string()
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            image_dir: default_image_dir(),
            log_level: default_log_level(),
        }
    }
}

/// Complete configuration
#[derive(Debug, Deserialize, Clone, Default)]
pub struct Config {
    #[serde(default)]
    pub ipc: IpcConfig,
    
    #[serde(default)]
    pub camera: CameraConfig,
    
    #[serde(default)]
    pub beam_shifter: BeamShifterConfig,
    
    #[serde(default)]
    pub stage: StageConfig,
    
    #[serde(default)]
    pub output: OutputConfig,
}

impl Config {
    /// Load configuration from the default location (project root).
    pub fn load() -> Result<Self> {
        let config_path = find_config_file()?;
        Self::load_from(&config_path)
    }
    
    /// Load configuration from a specific path.
    pub fn load_from(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        
        let config: Config = toml::from_str(&contents)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))?;
        
        info!("Loaded configuration from {}", path.display());
        Ok(config)
    }
    
    /// Get the log level as a tracing Level.
    pub fn log_level(&self) -> tracing::Level {
        match self.output.log_level.to_lowercase().as_str() {
            "debug" => tracing::Level::DEBUG,
            "info" => tracing::Level::INFO,
            "warn" | "warning" => tracing::Level::WARN,
            "error" => tracing::Level::ERROR,
            _ => tracing::Level::INFO,
        }
    }
}

/// Find config.toml in SR_core/runner/.
fn find_config_file() -> Result<PathBuf> {
    // Try relative to current working directory: SR_core/runner/config.toml
    let cwd_config = PathBuf::from("SR_core/runner/config.toml");
    if cwd_config.exists() {
        return Ok(cwd_config);
    }
    
    // Try ../runner/config.toml (when running from SR_core/image_processing)
    let sibling_config = PathBuf::from("../runner/config.toml");
    if sibling_config.exists() {
        return Ok(sibling_config);
    }
    
    // Try from CARGO_MANIFEST_DIR (for development with cargo run)
    // CARGO_MANIFEST_DIR = SR_core/image_processing
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let manifest_path = PathBuf::from(&manifest_dir);
        
        // Go up to SR_core, then into runner
        if let Some(sr_core_dir) = manifest_path.parent() {
            let config_path = sr_core_dir.join("runner").join("config.toml");
            if config_path.exists() {
                return Ok(config_path);
            }
        }
    }
    
    // Try relative to executable location
    if let Ok(exe_path) = std::env::current_exe() {
        let mut current = exe_path.parent().map(|p| p.to_path_buf());
        
        for _ in 0..10 {
            if let Some(ref dir) = current {
                // Check for runner/config.toml
                let config_path = dir.join("runner").join("config.toml");
                if config_path.exists() {
                    return Ok(config_path);
                }
                
                // Check for SR_core/runner/config.toml
                let config_path = dir.join("SR_core").join("runner").join("config.toml");
                if config_path.exists() {
                    return Ok(config_path);
                }
                
                current = dir.parent().map(|p| p.to_path_buf());
            } else {
                break;
            }
        }
    }
    
    anyhow::bail!(
        "Could not find config.toml. Ensure it exists at SR_core/runner/config.toml"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.ipc.socket_path, "/tmp/sr_ipc.sock");
        assert_eq!(config.camera.resolution, [2048, 1536]);
        assert_eq!(config.beam_shifter.default_waveform, "manhattan");
    }

    #[test]
    fn test_parse_config() {
        let toml_str = r#"
[ipc]
socket_path = "/custom/socket.sock"

[camera]
frame_rate = 30

[beam_shifter]
default_waveform = "diamond"
"#;
        
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.ipc.socket_path, "/custom/socket.sock");
        assert_eq!(config.camera.frame_rate, 30);
        assert_eq!(config.beam_shifter.default_waveform, "diamond");
    }
}
