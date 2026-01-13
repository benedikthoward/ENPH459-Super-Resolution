"""
Entry point for running the shifting driver as a module.

Usage:
    uv run --package shifting-driver python -m shifting_driver
"""

from .controller import BeamShifterController, WaveformShape


def main():
    """Demo entry point."""
    print("Shifting Driver - Optotune XRP-20 Controller")
    print("=" * 50)
    print()
    print("This module provides control over the Optotune beam shifter.")
    print()
    print("Example usage:")
    print("  from shifting_driver import BeamShifterController, WaveformShape")
    print()
    print("  with BeamShifterController() as ctrl:")
    print("      ctrl.set_frame_rate(60)")
    print("      ctrl.set_waveform(WaveformShape.MANHATTAN)")
    print("      ctrl.start()")
    print("      # ... capture images ...")
    print("      ctrl.stop()")
    print()
    print("Note: Connect ICC-4C controller via USB before running.")


if __name__ == "__main__":
    main()
