//! Super Resolution Image Processing Server
//!
//! This binary handles:
//! - USB3 Vision camera capture (MER2-302-56U3M via GenICam)
//! - IPC communication with the Python shifting driver
//! - Super-resolution image reconstruction

mod ipc;

use anyhow::Result;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use crate::ipc::{DefaultHandler, IPCServer, DEFAULT_SOCKET_PATH};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::DEBUG)
        .with_target(false)
        .init();

    info!("Super Resolution Image Processing Server");
    info!("=========================================");

    // Create and start IPC server
    let mut server = IPCServer::new(DEFAULT_SOCKET_PATH);
    server.start().await?;

    // Use default handler for now
    // TODO: Replace with actual camera/processing handler
    let handler = DefaultHandler::default();
    
    info!("Server ready. Waiting for connections...");
    info!("Socket: {}", DEFAULT_SOCKET_PATH);
    
    // Run server (blocks forever)
    server.run(handler).await?;

    Ok(())
}
