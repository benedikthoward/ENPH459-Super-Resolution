//! Super Resolution Image Processing Server
//!
//! This binary handles:
//! - USB3 Vision camera capture (MER2-302-56U3M via GenICam)
//! - IPC communication with the Python shifting driver
//! - Super-resolution image reconstruction

mod config;
mod ipc;

use anyhow::Result;
use tracing::info;
use tracing_subscriber::FmtSubscriber;

use crate::config::Config;
use crate::ipc::{DefaultHandler, IPCServer};

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration
    let config = Config::load().unwrap_or_else(|e| {
        eprintln!("Warning: Could not load config.toml: {}. Using defaults.", e);
        Config::default()
    });
    
    // Initialize logging with configured level
    let subscriber = FmtSubscriber::builder()
        .with_max_level(config.log_level())
        .with_target(false)
        .init();

    info!("Super Resolution Image Processing Server");
    info!("=========================================");
    info!("Configuration:");
    info!("  Socket: {}", config.ipc.socket_path);
    info!("  Camera: {}x{} @ {}fps", 
        config.camera.resolution[0], 
        config.camera.resolution[1],
        config.camera.frame_rate
    );
    info!("  Log level: {}", config.output.log_level);

    // Create and start IPC server with configured socket path
    let mut server = IPCServer::new(&config.ipc.socket_path);
    server.start().await?;

    // Use default handler for now
    // TODO: Replace with actual camera/processing handler that uses config.camera
    let handler = DefaultHandler::default();
    
    info!("Server ready. Waiting for connections...");
    
    // Run server (blocks forever)
    server.run(handler).await?;

    Ok(())
}
