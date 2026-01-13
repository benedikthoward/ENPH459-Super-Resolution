#!/usr/bin/env python3
"""
Launcher script for the Super Resolution system.

Starts both the Rust image processing server and Python shifting driver
with proper process management and graceful shutdown.

Usage:
    uv run python SR_core/runner/launch.py
    uv run python SR_core/runner/launch.py --config custom_config.toml
    uv run python SR_core/runner/launch.py --rust-only
    uv run python SR_core/runner/launch.py --dry-run
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Optional


def find_sr_core_dir() -> Path:
    """Find the SR_core directory (parent of this script's directory)."""
    return Path(__file__).resolve().parent.parent


def find_project_root() -> Path:
    """Find the project root directory (parent of SR_core)."""
    return find_sr_core_dir().parent


def get_default_config_path() -> Path:
    """Get the default config.toml path (same directory as this script)."""
    return Path(__file__).resolve().parent / "config.toml"


def load_config(config_path: Path) -> dict:
    """Load configuration from TOML file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def wait_for_socket(socket_path: str, timeout: float = 10.0) -> bool:
    """Wait for the Unix socket to become available."""
    start_time = time.time()
    socket_file = Path(socket_path)
    
    while time.time() - start_time < timeout:
        if socket_file.exists():
            return True
        time.sleep(0.1)
    
    return False


class ProcessManager:
    """Manages the Rust server and Python driver processes."""
    
    def __init__(self, sr_core_dir: Path, config: dict, dry_run: bool = False):
        self.sr_core_dir = sr_core_dir
        self.project_root = sr_core_dir.parent
        self.config = config
        self.dry_run = dry_run
        self.rust_process: Optional[subprocess.Popen] = None
        self.python_process: Optional[subprocess.Popen] = None
        self._shutdown_requested = False
    
    def start_rust_server(self) -> bool:
        """Start the Rust image processing server."""
        rust_dir = self.sr_core_dir / "image_processing"
        
        if not rust_dir.exists():
            print(f"Error: Rust project directory not found: {rust_dir}")
            return False
        
        cmd = ["cargo", "run", "--release"]
        
        print(f"Starting Rust server...")
        print(f"  Directory: {rust_dir}")
        print(f"  Command: {' '.join(cmd)}")
        
        if self.dry_run:
            print("  [DRY RUN] Would start Rust server")
            return True
        
        try:
            self.rust_process = subprocess.Popen(
                cmd,
                cwd=rust_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            return True
        except Exception as e:
            print(f"Error starting Rust server: {e}")
            return False
    
    def start_python_driver(self) -> bool:
        """Start the Python shifting driver."""
        cmd = [
            sys.executable, "-m", "shifting_driver"
        ]
        
        # Set up environment to find the shifting_driver module
        env = os.environ.copy()
        driver_src = self.sr_core_dir / "shifting_driver" / "src"
        
        # Add to PYTHONPATH
        python_path = env.get("PYTHONPATH", "")
        if python_path:
            env["PYTHONPATH"] = f"{driver_src}:{python_path}"
        else:
            env["PYTHONPATH"] = str(driver_src)
        
        print(f"Starting Python shifting driver...")
        print(f"  Command: {' '.join(cmd)}")
        
        if self.dry_run:
            print("  [DRY RUN] Would start Python driver")
            return True
        
        try:
            self.python_process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            return True
        except Exception as e:
            print(f"Error starting Python driver: {e}")
            return False
    
    def stream_output(self):
        """Stream output from both processes."""
        import selectors
        
        sel = selectors.DefaultSelector()
        
        if self.rust_process and self.rust_process.stdout:
            sel.register(self.rust_process.stdout, selectors.EVENT_READ, "rust")
        if self.python_process and self.python_process.stdout:
            sel.register(self.python_process.stdout, selectors.EVENT_READ, "python")
        
        try:
            while not self._shutdown_requested:
                # Check if processes are still running
                if self.rust_process and self.rust_process.poll() is not None:
                    print(f"\n[Rust server exited with code {self.rust_process.returncode}]")
                    break
                if self.python_process and self.python_process.poll() is not None:
                    print(f"\n[Python driver exited with code {self.python_process.returncode}]")
                    break
                
                events = sel.select(timeout=0.5)
                for key, _ in events:
                    line = key.fileobj.readline()
                    if line:
                        prefix = "[Rust]  " if key.data == "rust" else "[Python]"
                        print(f"{prefix} {line}", end="")
        except KeyboardInterrupt:
            pass
        finally:
            sel.close()
    
    def shutdown(self):
        """Gracefully shutdown all processes."""
        self._shutdown_requested = True
        print("\nShutting down...")
        
        # Terminate Python driver first
        if self.python_process and self.python_process.poll() is None:
            print("  Stopping Python driver...")
            self.python_process.terminate()
            try:
                self.python_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("  Force killing Python driver...")
                self.python_process.kill()
        
        # Then terminate Rust server
        if self.rust_process and self.rust_process.poll() is None:
            print("  Stopping Rust server...")
            self.rust_process.terminate()
            try:
                self.rust_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("  Force killing Rust server...")
                self.rust_process.kill()
        
        # Clean up socket file
        socket_path = self.config.get("ipc", {}).get("socket_path", "/tmp/sr_ipc.sock")
        socket_file = Path(socket_path)
        if socket_file.exists():
            try:
                socket_file.unlink()
                print(f"  Removed socket: {socket_path}")
            except Exception:
                pass
        
        print("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Launch the Super Resolution system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python SR_core/runner/launch.py              # Start full system
    uv run python SR_core/runner/launch.py --rust-only  # Start only Rust server
    uv run python SR_core/runner/launch.py --dry-run    # Show what would run
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file (default: SR_core/runner/config.toml)",
    )
    parser.add_argument(
        "--rust-only",
        action="store_true",
        help="Only start the Rust image processing server",
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Only start the Python shifting driver",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without starting processes",
    )
    
    args = parser.parse_args()
    
    # Find directories and load config
    sr_core_dir = find_sr_core_dir()
    project_root = find_project_root()
    config_path = args.config or get_default_config_path()
    
    print("=" * 60)
    print("Super Resolution System Launcher")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"SR_core dir:  {sr_core_dir}")
    print(f"Config file:  {config_path}")
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    socket_path = config.get("ipc", {}).get("socket_path", "/tmp/sr_ipc.sock")
    print(f"IPC socket:   {socket_path}")
    print()
    
    # Create process manager
    manager = ProcessManager(sr_core_dir, config, dry_run=args.dry_run)
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        manager.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start Rust server (unless python-only)
        if not args.python_only:
            if not manager.start_rust_server():
                sys.exit(1)
            
            if not args.dry_run:
                # Wait for socket to be ready
                print(f"Waiting for IPC socket...")
                if not wait_for_socket(socket_path, timeout=30):
                    print(f"Warning: Socket {socket_path} not available after 30s")
                else:
                    print(f"Socket ready: {socket_path}")
        
        # Start Python driver (unless rust-only)
        if not args.rust_only:
            print()
            if not manager.start_python_driver():
                manager.shutdown()
                sys.exit(1)
        
        if args.dry_run:
            print("\n[DRY RUN] Would now stream output from processes")
            print("[DRY RUN] Press Ctrl+C to exit")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            # Stream output
            print()
            print("-" * 60)
            print("Streaming process output (Ctrl+C to stop)")
            print("-" * 60)
            manager.stream_output()
    
    finally:
        manager.shutdown()


if __name__ == "__main__":
    main()
