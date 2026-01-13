# ENPH459-Super-Resolution

UBC Engineering Physics capstone investigating barcode decoding improvements using Optotune beam shifted images.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SR_core (Production)                           │
│  ┌──────────────────────────┐       ┌──────────────────────────────────┐   │
│  │   shifting_driver (Py)   │       │     image_processing (Rust)      │   │
│  │                          │       │                                  │   │
│  │  ┌────────────────────┐  │       │  ┌────────────┐  ┌───────────┐  │   │
│  │  │   optoICC SDK      │  │ Unix  │  │  Camera    │  │    SR     │  │   │
│  │  │   XPR Control      │◄─┼─Socket┼─►│  Driver    │  │ Algorithm │  │   │
│  │  └────────────────────┘  │       │  └────────────┘  └───────────┘  │   │
│  └──────────────────────────┘       └──────────────────────────────────┘   │
│              │                                    │                         │
└──────────────┼────────────────────────────────────┼─────────────────────────┘
               │                                    │
               ▼                                    ▼
        ┌─────────────┐                    ┌───────────────┐
        │   ICC-4C    │                    │ MER2-302-56U3M│
        │ Controller  │                    │    Camera     │
        └──────┬──────┘                    └───────────────┘
               │                                USB 3.0
               ▼
        ┌─────────────┐
        │   XRP-20    │
        │ Beam Shifter│
        └─────────────┘
```

## Hardware

| Component | Model | Interface | Notes |
|-----------|-------|-----------|-------|
| Beam Shifter | Optotune XRP-20 | Serial via ICC-4C | Manhattan/Diamond patterns, 50-120Hz |
| Controller | Optotune ICC-4C | USB Serial | 4-channel driver board |
| Camera | Daheng MER2-302-56U3M/C | USB3 Vision | Sony IMX265 CMOS, 3.2MP, GenICam compliant |

## Prerequisites

### macOS

```bash
# Install uv (Python package manager) and Rust
brew install uv rust
```

### Linux (Debian/Ubuntu)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# USB3 Vision camera support (optional, for aravis)
sudo apt install libusb-1.0-0-dev
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/ENPH459-Super-Resolution.git
cd ENPH459-Super-Resolution

# Install all Python dependencies (creates .venv automatically)
uv sync

# Launch the full system (Rust server + Python driver)
uv run python SR_core/runner/launch.py

# Or run components individually:
# Run prototyping scripts
uv run --package sr-prototyping python SR_prototyping/scripts/example.py

# Run shifting driver only
uv run --package shifting-driver python -m shifting_driver

# Build Rust image processor only
cd SR_core/image_processing
cargo build --release
```

## Project Structure

```
ENPH459-Super-Resolution/
├── pyproject.toml              # Root uv workspace config
├── .python-version             # Python 3.12
├── opt_materials/              # Optotune SDK and documentation
│   └── ICC-4C_PythonSDK_2.0.5256/
├── SR_prototyping/             # Algorithm R&D
│   ├── pyproject.toml
│   ├── scripts/                # Experiment scripts
│   └── resources/              # Test images, media
└── SR_core/                    # Production system
    ├── runner/                 # Launcher and shared config
    │   ├── config.toml         # Shared configuration
    │   └── launch.py           # System launcher script
    ├── shifting_driver/        # Python: Optotune control + IPC client
    │   ├── pyproject.toml
    │   └── src/shifting_driver/
    └── image_processing/       # Rust: Camera + SR algorithm + IPC server
        ├── Cargo.toml
        └── src/
```

## Hardware Setup

1. **ICC-4C Controller**: Connect via USB. Appears as a serial port (e.g., `/dev/ttyUSB0` on Linux, `/dev/cu.usbserial-*` on macOS, `COM3` on Windows).

2. **XRP-20 Beam Shifter**: Connect to ICC-4C channels 0-1. The XRP-20 uses two channels for X/Y beam deflection.

3. **MER2 Camera**: Connect to a USB 3.0 port (blue port). Ensure no USB hubs are between the camera and computer for full bandwidth.

## Development

### Prototyping (SR_prototyping)

Use this workspace for algorithm experimentation:

```bash
# Add a new dependency
uv add --package sr-prototyping scikit-image

# Run a script
uv run --package sr-prototyping python SR_prototyping/scripts/your_script.py
```

### Shifting Driver (SR_core/shifting_driver)

Python module for Optotune beam shifter control:

```bash
# Run the driver
uv run --package shifting-driver python -m shifting_driver

# Interactive Python with SDK
uv run --package shifting-driver python
>>> from shifting_driver import BeamShifterController
>>> ctrl = BeamShifterController()
```

### Image Processing (SR_core/image_processing)

Rust binary for camera capture and super-resolution:

```bash
cd SR_core/image_processing
cargo run --release
```

### Running the Full System

Use the launcher script to start both components:

```bash
# Start full system (Rust server + Python driver)
uv run python SR_core/runner/launch.py

# Start only the Rust server
uv run python SR_core/runner/launch.py --rust-only

# Start only the Python driver
uv run python SR_core/runner/launch.py --python-only

# Dry run (show what would be executed)
uv run python SR_core/runner/launch.py --dry-run

# Use a custom config file
uv run python SR_core/runner/launch.py --config /path/to/config.toml
```

## Configuration

Shared configuration is stored in `SR_core/runner/config.toml`:

```toml
[ipc]
socket_path = "/tmp/sr_ipc.sock"
timeout_seconds = 5.0

[camera]
device = "0"  # Device index or serial number
resolution = [2048, 1536]
frame_rate = 56
pixel_format = "Mono8"
exposure_us = 10000
gain_db = 0.0

[beam_shifter]
default_frame_rate = 60
default_waveform = "manhattan"
channel = 0
port = ""  # Empty for auto-detection

[output]
image_dir = "./output/frames"
log_level = "info"
```

## IPC Protocol

The Python shifting driver and Rust image processor communicate via Unix domain socket (`/tmp/sr_ipc.sock`) using JSON messages:

**Python → Rust:**
```json
{"type": "shift_start", "frame_rate": 60, "waveform": "manhattan"}
{"type": "shift_stop"}
{"type": "capture_request", "frames": 4}
{"type": "status"}
```

**Rust → Python:**
```json
{"type": "ack", "status": "ok"}
{"type": "capture_complete", "frame_ids": [1, 2, 3, 4]}
{"type": "error", "message": "Camera not connected"}
```
