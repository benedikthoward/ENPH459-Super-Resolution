#!/usr/bin/env python3
"""
Zaber Stage Jog Control
=======================
Interactive keyboard control for the Zaber XYZ stage.

Usage:
    uv run python SR_prototyping/zaber_setup/jog.py /dev/ttyUSB0

Controls:
    W / S   - Y axis  + / -
    A / D   - X axis  - / +
    Q / E   - Z axis  + / -
    [ / ]   - Decrease / Increase step size
    H       - Home all axes
    ESC     - Quit
"""

import curses
import sys

from zaber_motion import Units
from zaber_motion.ascii import Connection

STEP_SIZES_MM = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
DEFAULT_STEP_IDX = 2  # 0.1 mm


def _get_limit(axis, setting: str, fallback: float) -> float:
    try:
        return axis.settings.get(setting, Units.LENGTH_MILLIMETRES)
    except Exception:
        return fallback


def _get_pos(axis) -> float | None:
    try:
        return axis.get_position(Units.LENGTH_MILLIMETRES)
    except Exception:
        return None


def _move_clamped(axis, delta_mm: float, lim_min: float, lim_max: float) -> str | None:
    """Move relative, clamped to [lim_min, lim_max] to avoid out-of-range errors."""
    try:
        current = axis.get_position(Units.LENGTH_MILLIMETRES)
        target = max(lim_min, min(lim_max, current + delta_mm))
        actual_delta = target - current
        if abs(actual_delta) < 1e-4:
            return "At limit"
        axis.move_absolute(target, Units.LENGTH_MILLIMETRES, wait_until_idle=True)
    except Exception as e:
        return str(e)
    return None


def _home_all(x_axis, y_axis, z_axis) -> str | None:
    try:
        x_axis.home(wait_until_idle=True)
        y_axis.home(wait_until_idle=True)
        z_axis.home(wait_until_idle=True)
    except Exception as e:
        return str(e)
    return None


def _axis_bar(pos: float | None, lim_min: float, lim_max: float, width: int = 20) -> str:
    """Render a small ASCII progress bar showing position within limits."""
    if pos is None or lim_max <= lim_min:
        return "[" + "?" * width + "]"
    frac = (pos - lim_min) / (lim_max - lim_min)
    filled = int(round(frac * width))
    filled = max(0, min(width, filled))
    return "[" + "=" * filled + "-" * (width - filled) + "]"


def _jog_ui(stdscr, port: str) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)

    step_idx = DEFAULT_STEP_IDX
    status = "Connected."

    with Connection.open_serial_port(port) as connection:
        connection.enable_alerts()
        device_list = connection.detect_devices()
        if not device_list:
            raise RuntimeError("No Zaber devices found on port " + port)

        device = device_list[0]

        # Mirror the axis setup from main.py
        try:
            x_axis = device.get_lockstep(1)
            # Limits must be queried from a physical axis; both sides of lockstep share the same travel
            x_phys = device.get_axis(1)
        except Exception:
            x_axis = device.get_axis(1)
            x_phys = x_axis

        y_axis = device.get_axis(3)
        z_axis = device.get_axis(4)

        # Query limits once at startup
        x_min = _get_limit(x_phys, "limit.min", 0.0)
        x_max = _get_limit(x_phys, "limit.max", 100.0)
        y_min = _get_limit(y_axis, "limit.min", 0.0)
        y_max = _get_limit(y_axis, "limit.max", 100.0)
        z_min = _get_limit(z_axis, "limit.min", 0.0)
        z_max = _get_limit(z_axis, "limit.max", 100.0)

        # Warn if any axis is not homed — position will be unreliable until homed
        def _is_homed(axis) -> bool:
            try:
                return axis.is_homed()
            except Exception:
                return True  # assume homed if we can't tell

        needs_home = not (_is_homed(x_phys) and _is_homed(y_axis) and _is_homed(z_axis))
        if needs_home:
            status = "WARNING: axes not homed — press H to home before moving"

        while True:
            step = STEP_SIZES_MM[step_idx]
            x_pos = _get_pos(x_axis)
            y_pos = _get_pos(y_axis)
            z_pos = _get_pos(z_axis)

            def fmt(pos):
                return f"{pos:7.3f}" if pos is not None else "    N/A"

            stdscr.clear()
            stdscr.addstr(0, 0, "=== Zaber Stage Jog Control ===", curses.A_BOLD)
            stdscr.addstr(1, 0, f"Port : {port}")

            stdscr.addstr(3, 0, "Axis   Pos (mm)   Min      Max      Range")
            stdscr.addstr(4, 0, f"  X  {fmt(x_pos)}  {x_min:7.3f}  {x_max:7.3f}  {_axis_bar(x_pos, x_min, x_max)}")
            stdscr.addstr(5, 0, f"  Y  {fmt(y_pos)}  {y_min:7.3f}  {y_max:7.3f}  {_axis_bar(y_pos, y_min, y_max)}")
            stdscr.addstr(6, 0, f"  Z  {fmt(z_pos)}  {z_min:7.3f}  {z_max:7.3f}  {_axis_bar(z_pos, z_min, z_max)}")

            stdscr.addstr(8,  0, f"Step : {step:.3f} mm")
            stdscr.addstr(10, 0, "Controls:", curses.A_UNDERLINE)
            stdscr.addstr(11, 0, "  W / S   Y axis  + / -")
            stdscr.addstr(12, 0, "  A / D   X axis  - / +")
            stdscr.addstr(13, 0, "  Q / E   Z axis  + / -")
            stdscr.addstr(14, 0, "  [ / ]   Step size  smaller / larger")
            stdscr.addstr(15, 0, "  H       Home all axes")
            stdscr.addstr(16, 0, "  ESC     Quit")
            stdscr.addstr(18, 0, f"Status: {status}")
            stdscr.refresh()

            key = stdscr.getch()
            if key == -1:
                continue

            status = ""

            if key == 27:  # ESC
                break
            elif key in (ord('w'), ord('W')):
                err = _move_clamped(y_axis, +step, y_min, y_max)
                status = err or f"Y +{step:.3f} mm"
            elif key in (ord('s'), ord('S')):
                err = _move_clamped(y_axis, -step, y_min, y_max)
                status = err or f"Y -{step:.3f} mm"
            elif key in (ord('d'), ord('D')):
                err = _move_clamped(x_axis, +step, x_min, x_max)
                status = err or f"X +{step:.3f} mm"
            elif key in (ord('a'), ord('A')):
                err = _move_clamped(x_axis, -step, x_min, x_max)
                status = err or f"X -{step:.3f} mm"
            elif key in (ord('e'), ord('E')):
                err = _move_clamped(z_axis, +step, z_min, z_max)
                status = err or f"Z +{step:.3f} mm"
            elif key in (ord('q'), ord('Q')):
                err = _move_clamped(z_axis, -step, z_min, z_max)
                status = err or f"Z -{step:.3f} mm"
            elif key == ord(']'):
                step_idx = min(step_idx + 1, len(STEP_SIZES_MM) - 1)
                status = f"Step -> {STEP_SIZES_MM[step_idx]:.3f} mm"
            elif key == ord('['):
                step_idx = max(step_idx - 1, 0)
                status = f"Step -> {STEP_SIZES_MM[step_idx]:.3f} mm"
            elif key in (ord('h'), ord('H')):
                status = "Homing..."
                stdscr.addstr(18, 0, f"Status: {status}    ")
                stdscr.refresh()
                err = _home_all(x_axis, y_axis, z_axis)
                status = err or "Homed."


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        print("Error: serial port required.")
        print("  e.g.  uv run python SR_prototyping/zaber_setup/jog.py /dev/ttyUSB0")
        sys.exit(1)

    port = sys.argv[1]
    curses.wrapper(_jog_ui, port)


if __name__ == "__main__":
    main()
