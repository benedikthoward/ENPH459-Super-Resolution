//! IPC Server for communication with the Python shifting driver.
//!
//! Uses Unix domain sockets with JSON message protocol.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tracing::{debug, error, info, warn};

/// Default socket path
pub const DEFAULT_SOCKET_PATH: &str = "/tmp/sr_ipc.sock";

/// Incoming message types from Python client
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    ShiftStart {
        frame_rate: u32,
        waveform: String,
    },
    ShiftStop,
    CaptureRequest {
        frames: u32,
    },
    Status,
}

/// Outgoing message types to Python client
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    Ack {
        status: String,
    },
    CaptureComplete {
        frame_ids: Vec<u32>,
    },
    Error {
        message: String,
    },
    StatusResponse {
        is_running: bool,
        frame_rate: u32,
        frames_captured: u64,
    },
}

impl ServerMessage {
    /// Create an OK acknowledgment
    pub fn ok() -> Self {
        ServerMessage::Ack {
            status: "ok".to_string(),
        }
    }

    /// Create an error response
    pub fn error(message: impl Into<String>) -> Self {
        ServerMessage::Error {
            message: message.into(),
        }
    }
}

/// IPC Server state
pub struct IPCServer {
    socket_path: String,
    listener: Option<UnixListener>,
}

impl IPCServer {
    /// Create a new IPC server
    pub fn new(socket_path: impl Into<String>) -> Self {
        Self {
            socket_path: socket_path.into(),
            listener: None,
        }
    }

    /// Start listening for connections
    pub async fn start(&mut self) -> Result<()> {
        let path = Path::new(&self.socket_path);

        // Remove existing socket file if present
        if path.exists() {
            std::fs::remove_file(path)
                .context("Failed to remove existing socket file")?;
        }

        let listener = UnixListener::bind(path)
            .context("Failed to bind Unix socket")?;
        
        info!("IPC server listening on {}", self.socket_path);
        self.listener = Some(listener);
        Ok(())
    }

    /// Accept and handle incoming connections
    pub async fn run<H>(&self, mut handler: H) -> Result<()>
    where
        H: MessageHandler,
    {
        let listener = self.listener.as_ref()
            .context("Server not started. Call start() first.")?;

        loop {
            match listener.accept().await {
                Ok((stream, _addr)) => {
                    info!("Client connected");
                    if let Err(e) = handle_connection(stream, &mut handler).await {
                        error!("Connection error: {}", e);
                    }
                    info!("Client disconnected");
                }
                Err(e) => {
                    error!("Accept error: {}", e);
                }
            }
        }
    }
}

impl Drop for IPCServer {
    fn drop(&mut self) {
        // Clean up socket file
        let path = Path::new(&self.socket_path);
        if path.exists() {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Trait for handling incoming messages
pub trait MessageHandler {
    /// Handle shift start command
    fn on_shift_start(&mut self, frame_rate: u32, waveform: &str) -> ServerMessage;
    
    /// Handle shift stop command
    fn on_shift_stop(&mut self) -> ServerMessage;
    
    /// Handle capture request
    fn on_capture_request(&mut self, frames: u32) -> ServerMessage;
    
    /// Handle status request
    fn on_status(&mut self) -> ServerMessage;
}

/// Handle a single client connection
async fn handle_connection<H: MessageHandler>(
    stream: UnixStream,
    handler: &mut H,
) -> Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line).await?;
        
        if bytes_read == 0 {
            // Client disconnected
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        debug!("Received: {}", trimmed);

        // Parse and handle message
        let response = match serde_json::from_str::<ClientMessage>(trimmed) {
            Ok(msg) => match msg {
                ClientMessage::ShiftStart { frame_rate, waveform } => {
                    handler.on_shift_start(frame_rate, &waveform)
                }
                ClientMessage::ShiftStop => {
                    handler.on_shift_stop()
                }
                ClientMessage::CaptureRequest { frames } => {
                    handler.on_capture_request(frames)
                }
                ClientMessage::Status => {
                    handler.on_status()
                }
            },
            Err(e) => {
                warn!("Failed to parse message: {}", e);
                ServerMessage::error(format!("Invalid message: {}", e))
            }
        };

        // Send response
        let response_json = serde_json::to_string(&response)?;
        debug!("Sending: {}", response_json);
        writer.write_all(response_json.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;
    }

    Ok(())
}

/// Default message handler for testing
pub struct DefaultHandler {
    is_running: bool,
    frame_rate: u32,
    frames_captured: u64,
}

impl Default for DefaultHandler {
    fn default() -> Self {
        Self {
            is_running: false,
            frame_rate: 60,
            frames_captured: 0,
        }
    }
}

impl MessageHandler for DefaultHandler {
    fn on_shift_start(&mut self, frame_rate: u32, waveform: &str) -> ServerMessage {
        info!("Starting shift: {}Hz, {}", frame_rate, waveform);
        self.is_running = true;
        self.frame_rate = frame_rate;
        ServerMessage::ok()
    }

    fn on_shift_stop(&mut self) -> ServerMessage {
        info!("Stopping shift");
        self.is_running = false;
        ServerMessage::ok()
    }

    fn on_capture_request(&mut self, frames: u32) -> ServerMessage {
        info!("Capture request: {} frames", frames);
        
        // Generate mock frame IDs
        let start_id = self.frames_captured as u32;
        let frame_ids: Vec<u32> = (start_id..start_id + frames).collect();
        self.frames_captured += frames as u64;
        
        ServerMessage::CaptureComplete { frame_ids }
    }

    fn on_status(&mut self) -> ServerMessage {
        ServerMessage::StatusResponse {
            is_running: self.is_running,
            frame_rate: self.frame_rate,
            frames_captured: self.frames_captured,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_shift_start() {
        let json = r#"{"type": "shift_start", "frame_rate": 60, "waveform": "manhattan"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        
        match msg {
            ClientMessage::ShiftStart { frame_rate, waveform } => {
                assert_eq!(frame_rate, 60);
                assert_eq!(waveform, "manhattan");
            }
            _ => panic!("Expected ShiftStart"),
        }
    }

    #[test]
    fn test_serialize_response() {
        let response = ServerMessage::ok();
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"type\":\"ack\""));
        assert!(json.contains("\"status\":\"ok\""));
    }
}
