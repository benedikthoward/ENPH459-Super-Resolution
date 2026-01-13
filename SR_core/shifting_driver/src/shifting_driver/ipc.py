"""
IPC Client for communication with the Rust image processing server.

Uses Unix domain sockets with JSON message protocol.
"""

import json
import socket
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .config import get_config


class MessageType(Enum):
    """IPC message types."""
    # Client -> Server
    SHIFT_START = "shift_start"
    SHIFT_STOP = "shift_stop"
    CAPTURE_REQUEST = "capture_request"
    STATUS = "status"
    
    # Server -> Client
    ACK = "ack"
    CAPTURE_COMPLETE = "capture_complete"
    ERROR = "error"


@dataclass
class ShiftStartMessage:
    """Request to start beam shifting."""
    type: str = "shift_start"
    frame_rate: int = 60
    waveform: str = "manhattan"  # "manhattan" or "diamond"


@dataclass
class ShiftStopMessage:
    """Request to stop beam shifting."""
    type: str = "shift_stop"


@dataclass
class CaptureRequestMessage:
    """Request to capture frames."""
    type: str = "capture_request"
    frames: int = 4


@dataclass
class StatusMessage:
    """Request current status."""
    type: str = "status"


@dataclass
class Response:
    """Response from the server."""
    type: str
    status: Optional[str] = None
    message: Optional[str] = None
    frame_ids: Optional[list[int]] = None
    data: Optional[dict] = None
    
    @property
    def is_ok(self) -> bool:
        """Check if response indicates success."""
        return self.type == "ack" and self.status == "ok"
    
    @property
    def is_error(self) -> bool:
        """Check if response indicates error."""
        return self.type == "error"


class IPCClient:
    """
    IPC client for communicating with the Rust image processing server.
    
    Uses Unix domain sockets with newline-delimited JSON messages.
    
    Example:
        >>> client = IPCClient()
        >>> client.connect()
        >>> response = client.send_shift_start(frame_rate=60, waveform="manhattan")
        >>> if response.is_ok:
        ...     print("Shifting started")
        >>> client.disconnect()
    """
    
    def __init__(self, socket_path: Optional[Path] = None):
        """
        Initialize IPC client.
        
        Args:
            socket_path: Path to Unix domain socket. If None, uses config.toml setting.
        """
        if socket_path is None:
            config = get_config()
            socket_path = Path(config.ipc.socket_path)
        self._socket_path = socket_path
        self._timeout = get_config().ipc.timeout_seconds
        self._socket: Optional[socket.socket] = None
        self._is_connected = False
    
    def connect(self, timeout: Optional[float] = None) -> None:
        """
        Connect to the IPC server.
        
        Args:
            timeout: Connection timeout in seconds. If None, uses config.toml setting.
            
        Raises:
            ConnectionError: If unable to connect.
        """
        if self._is_connected:
            return
        
        if timeout is None:
            timeout = self._timeout
            
        try:
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.settimeout(timeout)
            self._socket.connect(str(self._socket_path))
            self._is_connected = True
        except socket.error as e:
            self._socket = None
            raise ConnectionError(
                f"Failed to connect to IPC server at {self._socket_path}: {e}"
            ) from e
    
    def disconnect(self) -> None:
        """Close connection to the server."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        self._is_connected = False
    
    def __enter__(self) -> "IPCClient":
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._is_connected
    
    def _send_message(self, message: dict[str, Any]) -> Response:
        """
        Send a message and receive response.
        
        Args:
            message: Message dictionary to send.
            
        Returns:
            Response from server.
            
        Raises:
            RuntimeError: If not connected.
            ConnectionError: If communication fails.
        """
        if not self._is_connected or not self._socket:
            raise RuntimeError("Not connected to IPC server. Call connect() first.")
        
        try:
            # Send newline-delimited JSON
            data = json.dumps(message) + "\n"
            self._socket.sendall(data.encode("utf-8"))
            
            # Receive response (read until newline)
            response_data = b""
            while True:
                chunk = self._socket.recv(4096)
                if not chunk:
                    raise ConnectionError("Server closed connection")
                response_data += chunk
                if b"\n" in response_data:
                    break
            
            # Parse response
            response_json = json.loads(response_data.decode("utf-8").strip())
            return Response(
                type=response_json.get("type", "unknown"),
                status=response_json.get("status"),
                message=response_json.get("message"),
                frame_ids=response_json.get("frame_ids"),
                data=response_json,
            )
            
        except json.JSONDecodeError as e:
            raise ConnectionError(f"Invalid JSON response: {e}") from e
        except socket.error as e:
            self._is_connected = False
            raise ConnectionError(f"Communication error: {e}") from e
    
    def send_shift_start(
        self,
        frame_rate: int = 60,
        waveform: str = "manhattan"
    ) -> Response:
        """
        Request the server to start beam shifting.
        
        Args:
            frame_rate: Shifting frame rate in Hz.
            waveform: "manhattan" or "diamond".
            
        Returns:
            Server response.
        """
        msg = ShiftStartMessage(frame_rate=frame_rate, waveform=waveform)
        return self._send_message(asdict(msg))
    
    def send_shift_stop(self) -> Response:
        """
        Request the server to stop beam shifting.
        
        Returns:
            Server response.
        """
        msg = ShiftStopMessage()
        return self._send_message(asdict(msg))
    
    def send_capture_request(self, frames: int = 4) -> Response:
        """
        Request the server to capture frames.
        
        Args:
            frames: Number of frames to capture.
            
        Returns:
            Server response with frame_ids on success.
        """
        msg = CaptureRequestMessage(frames=frames)
        return self._send_message(asdict(msg))
    
    def send_status(self) -> Response:
        """
        Request current status from server.
        
        Returns:
            Server response with status data.
        """
        msg = StatusMessage()
        return self._send_message(asdict(msg))
