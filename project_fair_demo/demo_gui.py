"""
Project Fair Demo GUI for Super-Resolution Imaging System.

Provides:
  1. Translation stage control (X/Y/Z jog buttons)
  2. Autofocus via laplacian variance maximization
  3. Capture with selectable shift pattern and half/full pixel mode
  4. SAA and SAA+IBP super-resolution reconstruction with live comparison

Run:
    uv run python -m project_fair_demo.demo_gui [--zaber-port PORT] [--xpr-port PORT]
"""

import sys
import time
import csv
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QPushButton, QDoubleSpinBox,
    QSpinBox, QComboBox, QProgressBar, QSplitter, QFrame,
    QSizePolicy, QButtonGroup, QRadioButton, QStatusBar, QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QPixmap, QImage

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── Project imports ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

CALIBRATION_DIR = ROOT / "stability_data" / "20260327_121155"
CALIBRATION_CSV = CALIBRATION_DIR / "shifts.csv"
PSF_CALIB_DIR   = ROOT / "stability_data" / "20260326_152815"

# Nominal XPR tilt values
NOMINAL_TILT_MONO = 0.14391
NOMINAL_TILT_COLOR = 0.28782

# 4-corner pattern signs and labels
CORNER_SIGNS = [(-1, +1), (+1, +1), (-1, -1), (+1, -1)]
CORNER_LABELS = ["(-x,+y)", "(+x,+y)", "(-x,-y)", "(+x,-y)"]
CORNER_TO_CAL_POS = {0: 0, 1: 2, 2: 6, 3: 8}

# Default hardware parameters
DEFAULT_SETTLE_MS = 50
TRIGGER_PULSE_US = 100
DEFAULT_GAIN = 0
UPSAMPLE_FACTOR = 2


# ═══════════════════════════════════════════════════════════════════════════
# Shift patterns: name -> list of (sign_x, sign_y) for each capture position
# ═══════════════════════════════════════════════════════════════════════════

SHIFT_PATTERNS = {
    "4 Corners": [(-1, +1), (+1, +1), (-1, -1), (+1, -1)],
    "Diamond": [(0, +1), (+1, 0), (0, -1), (-1, 0)],
    "3x3 Grid": [(sx, sy) for sy in [1, 0, -1] for sx in [-1, 0, 1]],
}


# ═══════════════════════════════════════════════════════════════════════════
# Calibration helpers (adapted from collect_raw_images_hw)
# ═══════════════════════════════════════════════════════════════════════════

def load_calibration(csv_path: str) -> dict:
    cal = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row["sweep_axis"], row["tilt_angle_deg"], int(row["position"]))
            cal[key] = (float(row["dx_mean_px"]), float(row["dy_mean_px"]))
    return cal


def interpolate_tilt_for_shift(csv_path: str, target_px: float) -> tuple[float, float]:
    """Interpolate (tilt_x, tilt_y) for a target pixel shift using centre positions."""
    tilts_x, shifts_x = [], []
    tilts_y, shifts_y = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            pos = int(row["position"])
            tilt = float(row["tilt_angle_deg"])
            if row["sweep_axis"] == "x" and pos == 5:
                tilts_x.append(tilt)
                shifts_x.append(abs(float(row["dx_mean_px"])))
            elif row["sweep_axis"] == "y" and pos == 7:
                tilts_y.append(tilt)
                shifts_y.append(abs(float(row["dy_mean_px"])))

    ox = np.argsort(shifts_x)
    tilt_x = float(np.interp(target_px, np.array(shifts_x)[ox], np.array(tilts_x)[ox]))
    oy = np.argsort(shifts_y)
    tilt_y = float(np.interp(target_px, np.array(shifts_y)[oy], np.array(tilts_y)[oy]))
    return tilt_x, tilt_y


def interpolate_tilt_for_corner(csv_path: str, target_px: float,
                                corner_idx: int) -> tuple[float, float]:
    """Interpolate tilt_x and tilt_y for one corner to achieve target_px shift."""
    cal_pos = CORNER_TO_CAL_POS[corner_idx]
    tilts_x, shifts_x = [], []
    tilts_y, shifts_y = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            pos = int(row["position"])
            if pos != cal_pos:
                continue
            tilt = float(row["tilt_angle_deg"])
            if row["sweep_axis"] == "x":
                tilts_x.append(tilt)
                shifts_x.append(abs(float(row["dx_mean_px"])))
            elif row["sweep_axis"] == "y":
                tilts_y.append(tilt)
                shifts_y.append(abs(float(row["dy_mean_px"])))
    if not tilts_x or not tilts_y:
        raise ValueError(f"Missing calibration data for corner {corner_idx}")
    ox = np.argsort(shifts_x)
    tilt_x = float(np.interp(target_px, np.array(shifts_x)[ox], np.array(tilts_x)[ox]))
    oy = np.argsort(shifts_y)
    tilt_y = float(np.interp(target_px, np.array(shifts_y)[oy], np.array(tilts_y)[oy]))
    return tilt_x, tilt_y


# ═══════════════════════════════════════════════════════════════════════════
# Super-resolution algorithms
# Matches sweep_sr_barcodes_rgb_NEW.py exactly.
#
# Shifts are (dy, dx) in red-channel LR pixels.
# For RGGB with 1.0 sensor-pixel diagonal shift:
#   corner0 (-x,+y) → (dy, dx) = (+0.5, -0.5)
#   corner1 (+x,+y) → (dy, dx) = (+0.5, +0.5)
#   corner2 (-x,-y) → (dy, dx) = (-0.5, -0.5)
#   corner3 (+x,-y) → (dy, dx) = (-0.5, +0.5)
# ═══════════════════════════════════════════════════════════════════════════

from scipy.ndimage import shift as ndi_shift, zoom as ndi_zoom
from scipy.signal import fftconvolve

# Nominal (dy, dx) shifts in red-channel LR pixels for each corner
CORNER_SHIFTS_LR = [
    (+0.5, -0.5),  # corner0 (-x,+y)
    (+0.5, +0.5),  # corner1 (+x,+y)
    (-0.5, -0.5),  # corner2 (-x,-y)
    (-0.5, +0.5),  # corner3 (+x,-y)
]

PSF_HALFWIDTH = 3
IBP_ITERATIONS = 80
IBP_STEP_SIZE = 0.5


def make_gaussian_psf(halfwidth: int = PSF_HALFWIDTH) -> np.ndarray:
    """Create a small Gaussian PSF as fallback when no calibration exists."""
    size = 2 * halfwidth + 1
    c = halfwidth
    y, x = np.mgrid[:size, :size]
    psf = np.exp(-((x - c)**2 + (y - c)**2) / (2 * 1.0**2))
    psf /= psf.sum()
    return psf


def load_measured_psf(calib_dir: Path = PSF_CALIB_DIR,
                      halfwidth: int = PSF_HALFWIDTH) -> np.ndarray:
    """
    Load measured PSF from pos4_(0,0).png images in the calibration dir.

    Finds all sweep subdirectories, loads the pos4_(0,0).png from each,
    aligns by peak location, averages, and extracts a (2*halfwidth+1)^2 kernel.
    """
    from PIL import Image

    margin = halfwidth + 6
    patches = []
    used_files = []

    for sweep_dir in sorted(calib_dir.iterdir()):
        if not sweep_dir.is_dir():
            continue
        pos4_path = sweep_dir / "pos4_(0,0).png"
        if not pos4_path.exists():
            continue
        img = np.array(Image.open(pos4_path), dtype=np.float64)
        if img.ndim == 3:
            img = img.mean(axis=2)
        pr, pc = np.unravel_index(img.argmax(), img.shape)
        R = margin
        if (pr < R or pr + R + 1 > img.shape[0]
                or pc < R or pc + R + 1 > img.shape[1]):
            continue
        patches.append(img[pr - R:pr + R + 1, pc - R:pc + R + 1].copy())
        used_files.append(str(pos4_path))

    if not patches:
        raise FileNotFoundError(
            f"No pos4_(0,0).png found under {calib_dir}")

    avg = np.mean(patches, axis=0)
    R = margin
    kernel = avg[R - halfwidth:R + halfwidth + 1,
                 R - halfwidth:R + halfwidth + 1].copy()

    # Subtract corner background and normalise
    corners = np.concatenate([
        kernel[:3, :3].ravel(), kernel[:3, -3:].ravel(),
        kernel[-3:, :3].ravel(), kernel[-3:, -3:].ravel(),
    ])
    kernel -= np.mean(corners)
    kernel = np.clip(kernel, 0, None)
    kernel /= kernel.sum()

    print(f"PSF: averaged {len(patches)} pos4_(0,0).png images, "
          f"kernel shape {kernel.shape}")
    return kernel


# Cache so we only load once per session
_measured_psf_cache: dict[str, np.ndarray] = {}


def get_psf(use_measured: bool = True,
            calib_dir: Path = PSF_CALIB_DIR) -> np.ndarray:
    """Return either the measured or Gaussian PSF."""
    if not use_measured:
        return make_gaussian_psf()
    key = str(calib_dir)
    if key not in _measured_psf_cache:
        _measured_psf_cache[key] = load_measured_psf(calib_dir)
    return _measured_psf_cache[key]


def _blur(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return fftconvolve(img, kernel, mode='same')


def _forward_model(hr: np.ndarray, kernel: np.ndarray,
                   shift_yx: tuple[float, float], factor: int) -> np.ndarray:
    """HR → blur → shift → downsample → simulated LR."""
    blurred = _blur(hr, kernel)
    shifted = ndi_shift(blurred, (shift_yx[0] * factor, shift_yx[1] * factor),
                        order=3, mode='nearest')
    return shifted[::factor, ::factor]


def _back_project(error_lr: np.ndarray, kernel: np.ndarray,
                  shift_yx: tuple[float, float], factor: int,
                  hr_shape: tuple[int, int]) -> np.ndarray:
    """Back-project LR error into HR space."""
    h_hr, w_hr = hr_shape
    up = np.zeros((error_lr.shape[0] * factor, error_lr.shape[1] * factor))
    up[::factor, ::factor] = error_lr
    if up.shape[0] < h_hr or up.shape[1] < w_hr:
        up = np.pad(up, ((0, max(0, h_hr - up.shape[0])),
                         (0, max(0, w_hr - up.shape[1]))))
    up = up[:h_hr, :w_hr]
    shifted = ndi_shift(up, (-shift_yx[0] * factor, -shift_yx[1] * factor),
                        order=3, mode='nearest')
    return _blur(shifted, kernel[::-1, ::-1])


def shift_and_add(lr_list: list[np.ndarray],
                  shifts_yx: list[tuple[float, float]],
                  factor: int = 2) -> np.ndarray:
    """Shift-and-Add: upsample each LR, shift, average."""
    h_lr, w_lr = lr_list[0].shape
    acc = np.zeros((h_lr * factor, w_lr * factor), dtype=np.float64)
    for lr, (dy, dx) in zip(lr_list, shifts_yx):
        up = ndi_zoom(lr.astype(np.float64), factor, order=3)
        acc += ndi_shift(up, (dy * factor, dx * factor), order=3, mode='nearest')
    return acc / len(lr_list)


def ibp(lr_list: list[np.ndarray],
        shifts_yx: list[tuple[float, float]],
        kernel: np.ndarray,
        hr_init: np.ndarray,
        factor: int = 2,
        n_iter: int = IBP_ITERATIONS,
        step: float = IBP_STEP_SIZE,
        progress_cb=None) -> np.ndarray:
    """Iterative Back Projection. Returns final HR image."""
    hr = hr_init.copy().astype(np.float64)
    n = len(lr_list)
    lr_f64 = [f.astype(np.float64) for f in lr_list]

    for it in range(n_iter):
        correction = np.zeros_like(hr)
        total_err = 0.0
        for lr, s in zip(lr_f64, shifts_yx):
            sim = _forward_model(hr, kernel, s, factor)
            mh = min(sim.shape[0], lr.shape[0])
            mw = min(sim.shape[1], lr.shape[1])
            err = lr[:mh, :mw] - sim[:mh, :mw]
            total_err += np.mean(err ** 2)
            correction += _back_project(err, kernel, s, factor, hr.shape)
        hr += step * correction / n
        hr = np.clip(hr, 0, 255)
        mse = total_err / n
        if progress_cb:
            progress_cb(it + 1, n_iter, mse)

    return hr.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Pattern preview widget
# ═══════════════════════════════════════════════════════════════════════════

class PatternPreview(QWidget):
    """Draws dots showing the shift pattern positions."""

    clicked = pyqtSignal()

    def __init__(self, name: str, signs: list[tuple[int, int]], parent=None):
        super().__init__(parent)
        self.name = name
        self.signs = signs
        self._selected = False
        self.setFixedSize(90, 90)
        self.setCursor(Qt.PointingHandCursor)

    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, val: bool):
        self._selected = val
        self.update()

    def mousePressEvent(self, ev):
        self.clicked.emit()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Background
        if self._selected:
            p.fillRect(self.rect(), QColor("#2d5a8e"))
            p.setPen(QPen(QColor("#6cb4ee"), 2))
            p.drawRoundedRect(1, 1, self.width() - 2, self.height() - 2, 6, 6)
        else:
            p.fillRect(self.rect(), QColor("#1e1e2e"))
            p.setPen(QPen(QColor("#444466"), 1))
            p.drawRoundedRect(1, 1, self.width() - 2, self.height() - 2, 6, 6)

        # Grid lines
        cx, cy = self.width() // 2, self.height() // 2 - 4
        grid_r = 28
        p.setPen(QPen(QColor("#333355"), 1, Qt.DotLine))
        p.drawLine(cx - grid_r, cy, cx + grid_r, cy)
        p.drawLine(cx, cy - grid_r, cx, cy + grid_r)

        # Dots
        dot_color = QColor("#6cb4ee") if self._selected else QColor("#88aadd")
        p.setBrush(dot_color)
        p.setPen(Qt.NoPen)
        for sx, sy in self.signs:
            dx = cx + int(sx * grid_r * 0.8)
            dy = cy - int(sy * grid_r * 0.8)  # y inverted for screen
            p.drawEllipse(dx - 4, dy - 4, 8, 8)

        # Label
        p.setPen(QColor("#aaaacc"))
        p.setFont(QFont("sans-serif", 7))
        p.drawText(self.rect().adjusted(0, 0, 0, 2), Qt.AlignBottom | Qt.AlignHCenter,
                   self.name)
        p.end()


class ROIPreviewLabel(QLabel):
    """QLabel that displays an image and lets the user drag a rectangle ROI.
    The ROI is stored in *image* coordinates (before display scaling).
    """

    roi_changed = pyqtSignal(int, int, int, int)  # x, y, w, h in image coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._img_w = 0     # original image dimensions
        self._img_h = 0
        # ROI in image coordinates (x, y, w, h)
        self._roi = None
        self._dragging = False
        self._drag_start = None
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    def set_image(self, pixmap: QPixmap, img_w: int, img_h: int):
        """Set the display pixmap and remember the original image size."""
        self._pixmap = pixmap
        self._img_w = img_w
        self._img_h = img_h
        self.update()

    @property
    def roi_image_coords(self):
        """Return (x, y, w, h) in original image coordinates, or None."""
        return self._roi

    def _display_rect(self):
        """Rect where the scaled image is actually drawn (centred)."""
        if not self._pixmap:
            return self.rect()
        pw = self._pixmap.width()
        ph = self._pixmap.height()
        lw, lh = self.width(), self.height()
        scale = min(lw / pw, lh / ph, 1.0) if pw and ph else 1.0
        dw, dh = int(pw * scale), int(ph * scale)
        dx = (lw - dw) // 2
        dy = (lh - dh) // 2
        return dx, dy, dw, dh, scale

    def _widget_to_image(self, wx, wy):
        dx, dy, dw, dh, scale = self._display_rect()
        if scale == 0:
            return 0, 0
        ix = int((wx - dx) / scale)
        iy = int((wy - dy) / scale)
        ix = max(0, min(ix, self._img_w - 1))
        iy = max(0, min(iy, self._img_h - 1))
        return ix, iy

    def _image_to_widget(self, ix, iy):
        dx, dy, dw, dh, scale = self._display_rect()
        return int(ix * scale + dx), int(iy * scale + dy)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._img_w:
            self._dragging = True
            self._drag_start = (ev.x(), ev.y())

    def mouseMoveEvent(self, ev):
        pass  # could show crosshair, but keeping simple

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            x0, y0 = self._widget_to_image(*self._drag_start)
            x1, y1 = self._widget_to_image(ev.x(), ev.y())
            rx = min(x0, x1)
            ry = min(y0, y1)
            rw = abs(x1 - x0)
            rh = abs(y1 - y0)
            if rw > 10 and rh > 10:
                self._roi = (rx, ry, rw, rh)
                self.roi_changed.emit(rx, ry, rw, rh)
                self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0a0a15"))
        if self._pixmap:
            dx, dy, dw, dh, scale = self._display_rect()
            p.drawPixmap(dx, dy, dw, dh, self._pixmap)
            # Draw ROI rectangle
            if self._roi:
                rx, ry, rw, rh = self._roi
                wx, wy = self._image_to_widget(rx, ry)
                ww = int(rw * scale)
                wh = int(rh * scale)
                p.setPen(QPen(QColor("#ff4444"), 2, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawRect(wx, wy, ww, wh)
                # Label
                p.setPen(QColor("#ff6666"))
                p.setFont(QFont("sans-serif", 9))
                p.drawText(wx + 4, wy - 4, f"ROI {rw}x{rh}")
        p.end()


# ═══════════════════════════════════════════════════════════════════════════
# Worker threads
# ═══════════════════════════════════════════════════════════════════════════

class LiveFeedWorker(QThread):
    """Grabs frames at ~5 Hz for the live preview. Pauses when told to."""
    frame = pyqtSignal(object)  # numpy image (copied)

    def __init__(self, imaging_sys, is_color=False):
        super().__init__()
        self.sys = imaging_sys
        self.is_color = is_color
        self._running = True
        self._paused = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue
            try:
                raw = self.sys._capture_frame()
                if self.is_color:
                    img = cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR)
                else:
                    img = raw
                self.frame.emit(img.copy())
            except Exception:
                pass
            time.sleep(0.2)  # ~5 Hz


class ConnectWorker(QThread):
    """Runs all hardware initialisation off the main thread."""
    status_msg = pyqtSignal(str)
    finished = pyqtSignal(object, object, object, float, bool)
    #                      imaging_sys, zaber_conn, zaber_axis, auto_exp, is_color
    error = pyqtSignal(str)

    def __init__(self, xpr_port: str, zaber_port: str):
        super().__init__()
        self.xpr_port = xpr_port
        self.zaber_port = zaber_port

    def run(self):
        imaging_sys = None
        zaber_conn = None
        zaber_axis = None
        try:
            # 1. Auto-expose with a software-triggered camera, then close it
            #    (Daheng SDK allows only one connection at a time).
            self.status_msg.emit("Running auto-exposure...")
            from imaging.camera import DahengCamera
            cam_sw = DahengCamera(hardware_trigger=False)
            cam_sw.gain = DEFAULT_GAIN
            auto_exp = cam_sw.auto_exposure()
            is_color = cam_sw.is_color
            cam_sw.close()
            del cam_sw

            # 2. Open ImagingSystem (camera + XPR)
            self.status_msg.emit("Connecting to XPR + camera (hw trigger)...")
            from imaging import ImagingSystem
            imaging_sys = ImagingSystem(
                hardware_trigger=True,
                xpr_port=self.xpr_port or None,
            )
            imaging_sys.camera.gain = DEFAULT_GAIN
            imaging_sys.camera.exposure = auto_exp

            # 3. Zaber stage (optional)
            if self.zaber_port:
                self.status_msg.emit("Connecting to Zaber stage...")
                from zaber_motion import Units
                from zaber_motion.ascii import Connection
                zaber_conn = Connection.open_serial_port(self.zaber_port)
                zaber_conn.enable_alerts()
                devices = zaber_conn.detect_devices()
                if devices:
                    device = devices[0]
                    try:
                        zaber_axis = device.get_lockstep(1)
                    except Exception:
                        zaber_axis = device.get_axis(1)

            self.finished.emit(imaging_sys, zaber_conn, zaber_axis,
                               auto_exp, is_color)

        except Exception as e:
            # Clean up on failure
            if imaging_sys:
                try:
                    imaging_sys.close()
                except Exception:
                    pass
            if zaber_conn:
                try:
                    zaber_conn.close()
                except Exception:
                    pass
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class AutoExposureWorker(QThread):
    """Binary-search auto-exposure using the hw-triggered camera."""
    status_msg = pyqtSignal(str)
    preview = pyqtSignal(object)          # numpy image for live preview
    finished = pyqtSignal(float)          # final exposure_us
    error = pyqtSignal(str)

    TARGET_PEAK = 230
    MAX_ITER = 12

    def __init__(self, imaging_sys, is_color=False):
        super().__init__()
        self.sys = imaging_sys
        self.is_color = is_color

    def run(self):
        try:
            cam = self.sys.camera
            exp = cam.exposure
            # If current exposure is very low, start with a reasonable guess
            if exp < 100:
                exp = 5000.0

            for i in range(self.MAX_ITER):
                cam.exposure = exp
                time.sleep(0.05)
                raw = self.sys._capture_frame()
                peak = int(raw.max())
                self.status_msg.emit(
                    f"Auto-exposure iter {i+1}: exp={exp:.0f} us, peak={peak}/255")
                if self.is_color:
                    self.preview.emit(cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR).copy())
                else:
                    self.preview.emit(raw.copy())

                if abs(peak - self.TARGET_PEAK) <= 5:
                    break
                if peak == 0:
                    exp *= 4
                    exp = min(500000, exp)
                    continue
                exp = exp * (self.TARGET_PEAK / max(peak, 1))
                exp = max(100, min(500000, exp))

            cam.exposure = exp
            self.finished.emit(exp)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class AutofocusWorker(QThread):
    progress = pyqtSignal(int, int, float, float)   # step, total, pos_mm, lap_var
    preview = pyqtSignal(object)                     # numpy image at each step
    finished = pyqtSignal(float, list, list)         # best_pos, positions, variances
    error = pyqtSignal(str)

    def __init__(self, zaber_axis, imaging_sys, range_mm, step_mm, start_pos,
                 is_color=False):
        super().__init__()
        self.axis = zaber_axis
        self.sys = imaging_sys
        self.range_mm = range_mm
        self.step_mm = step_mm
        self.start_pos = start_pos
        self.is_color = is_color

    def run(self):
        try:
            from zaber_motion import Units
            offsets = np.arange(-self.range_mm, self.range_mm + self.step_mm / 2,
                                self.step_mm)
            positions = self.start_pos + offsets
            variances = []

            # Throw away the first capture — often garbage from stale buffer
            self.axis.move_absolute(positions[0], Units.LENGTH_MILLIMETRES,
                                    wait_until_idle=True)
            time.sleep(0.25)
            self.sys.xpr.set_home()
            time.sleep(0.02)
            self.sys._capture_frame()  # discard

            for i, pos in enumerate(positions):
                self.axis.move_absolute(pos, Units.LENGTH_MILLIMETRES,
                                        wait_until_idle=True)
                time.sleep(0.25)
                self.sys.xpr.set_home()
                time.sleep(0.02)
                raw = self.sys._capture_frame()
                # Demosaic if color camera (raw is Bayer RGGB)
                if self.is_color:
                    bgr = cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR)
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    self.preview.emit(bgr.copy())
                else:
                    gray = raw
                    self.preview.emit(raw.copy())
                lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                variances.append(lap)
                self.progress.emit(i + 1, len(positions), float(pos), lap)

            best_idx = int(np.argmax(variances))
            best_pos = float(positions[best_idx])

            # Move to best position
            self.axis.move_absolute(best_pos, Units.LENGTH_MILLIMETRES,
                                    wait_until_idle=True)

            self.finished.emit(best_pos, positions.tolist(), variances)

        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class CaptureWorker(QThread):
    progress = pyqtSignal(str)
    image_captured = pyqtSignal(int, object)  # index, numpy array
    finished = pyqtSignal(list, list)         # images, shifts_px
    error = pyqtSignal(str)

    def __init__(self, imaging_sys, positions, settle_ms, is_color):
        super().__init__()
        self.sys = imaging_sys
        self.positions = positions  # list of (angle_x, angle_y)
        self.settle_ms = settle_ms
        self.is_color = is_color

    def run(self):
        try:
            images = []
            for i, (ax, ay) in enumerate(self.positions):
                self.progress.emit(f"Capturing position {i+1}/{len(self.positions)} "
                                   f"({ax:+.5f}, {ay:+.5f})")
                self.sys.xpr.set_angles(ax, ay)
                time.sleep(self.settle_ms / 1000.0)
                img = self.sys._capture_frame()
                images.append(img)
                self.image_captured.emit(i, img.copy())

            self.sys.xpr.set_home()

            # Compute expected pixel shifts from tilt angles
            # For the demo, use calibration-based shifts
            shifts_px = []
            cal_path = str(CALIBRATION_CSV)
            if CALIBRATION_CSV.exists():
                cal = load_calibration(cal_path)
                for i, (ax, ay) in enumerate(self.positions):
                    # Use tilt magnitude to estimate pixel shift
                    tilt_x = abs(ax)
                    tilt_y = abs(ay)
                    sign_x = 1 if ax >= 0 else -1
                    sign_y = 1 if ay >= 0 else -1

                    # Look up nearest calibration entry for x-sweep pos 5
                    cal_tilts_x = sorted(set(
                        float(k[1]) for k in cal if k[0] == "x" and k[2] == 5
                    ))
                    if cal_tilts_x and tilt_x > 0:
                        closest = min(cal_tilts_x, key=lambda t: abs(t - tilt_x))
                        entry = cal.get(("x", f"{closest:.5f}", 5))
                        dx = entry[0] * sign_x if entry else 0.0
                    else:
                        dx = 0.0

                    cal_tilts_y = sorted(set(
                        float(k[1]) for k in cal if k[0] == "y" and k[2] == 7
                    ))
                    if cal_tilts_y and tilt_y > 0:
                        closest = min(cal_tilts_y, key=lambda t: abs(t - tilt_y))
                        entry = cal.get(("y", f"{closest:.5f}", 7))
                        dy = entry[1] * sign_y if entry else 0.0
                    else:
                        dy = 0.0

                    shifts_px.append((dx, dy))
            else:
                # Fallback: assume nominal shifts
                for i, (ax, ay) in enumerate(self.positions):
                    sx = 1 if ax >= 0 else -1
                    sy = 1 if ay >= 0 else -1
                    target = 1.0 if self.is_color else 0.5
                    shifts_px.append((sx * target, sy * target))

            self.finished.emit(images, shifts_px)

        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class ReconstructionWorker(QThread):
    progress = pyqtSignal(str)
    ibp_step = pyqtSignal(int, int, float)       # iteration, total, mse
    finished = pyqtSignal(object, object, object)  # original, saa, saa_ibp
    error = pyqtSignal(str)

    def __init__(self, images, roi=None, is_color=False, roi_only=True,
                 use_measured_psf=True):
        super().__init__()
        self.images = images
        self.roi = roi  # (x, y, w, h) in full-sensor image coords or None
        self.is_color = is_color
        self.roi_only = roi_only  # True = process only ROI, False = full then crop
        self.use_measured_psf = use_measured_psf

    def run(self):
        try:
            h, w = self.images[0].shape[:2]
            factor = UPSAMPLE_FACTOR

            # Extract red channel from RGGB Bayer: R at even rows, even cols
            if self.is_color:
                raw_reds = [img[0::2, 0::2].astype(np.float64) for img in self.images]
                rh, rw = raw_reds[0].shape
            else:
                raw_reds = [img.astype(np.float64) for img in self.images]
                rh, rw = h, w

            # Compute ROI in red-channel coords for display cropping
            if self.roi:
                rx, ry, roi_w, roi_h = self.roi
                if self.is_color:
                    rx, ry = rx // 2, ry // 2
                    roi_w, roi_h = roi_w // 2, roi_h // 2
            else:
                crop_size = min(256, rh, rw)
                rx = (rw - crop_size) // 2
                ry = (rh - crop_size) // 2
                roi_w = roi_h = crop_size

            if self.roi_only:
                # Process only the ROI crop
                lr_frames = [img[ry:ry+roi_h, rx:rx+roi_w].copy()
                             for img in raw_reds]
                roi_hr = None  # no separate crop needed
            else:
                # Process full image, will crop for display later
                lr_frames = [img.copy() for img in raw_reds]
                # HR-space ROI coords for cropping results
                roi_hr = (ry * factor, rx * factor,
                          roi_h * factor, roi_w * factor)

            shifts_yx = CORNER_SHIFTS_LR[:len(lr_frames)]
            original_lr = lr_frames[0].copy()

            self.progress.emit("Running SAA on red channel...")
            saa_hr = shift_and_add(lr_frames, shifts_yx, factor=factor)

            self.progress.emit(f"Running IBP ({IBP_ITERATIONS} iters) on red channel...")
            psf_kernel = get_psf(use_measured=self.use_measured_psf)
            saa_ibp = ibp(
                lr_frames, shifts_yx, psf_kernel, saa_hr.copy(),
                factor=factor, n_iter=IBP_ITERATIONS, step=IBP_STEP_SIZE,
                progress_cb=lambda i, t, mse: self.ibp_step.emit(i, t, mse),
            )

            # Crop to ROI for display if we processed full image
            if roi_hr is not None:
                cry, crx, crh, crw = roi_hr
                saa_hr = saa_hr[cry:cry+crh, crx:crx+crw]
                saa_ibp = saa_ibp[cry:cry+crh, crx:crx+crw]
                original_lr = original_lr[ry:ry+roi_h, rx:rx+roi_w]

            self.finished.emit(original_lr, saa_hr, saa_ibp)

        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════
# Main GUI
# ═══════════════════════════════════════════════════════════════════════════

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0f0f1a;
    color: #d0d0e0;
    font-family: 'Segoe UI', 'Ubuntu', sans-serif;
}
QGroupBox {
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    margin-top: 12px;
    padding: 14px 10px 10px 10px;
    font-weight: bold;
    font-size: 13px;
    color: #8899cc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
}
QPushButton {
    background-color: #1a1a3a;
    border: 1px solid #3a3a6a;
    border-radius: 6px;
    padding: 6px 14px;
    color: #c0c0e0;
    font-size: 12px;
    min-height: 28px;
}
QPushButton:hover {
    background-color: #252555;
    border-color: #5566aa;
}
QPushButton:pressed {
    background-color: #303070;
}
QPushButton:disabled {
    background-color: #111122;
    color: #444466;
    border-color: #222244;
}
QPushButton#accentBtn {
    background-color: #1a4a8a;
    border-color: #3377cc;
    color: #e0e8ff;
    font-weight: bold;
    font-size: 14px;
    min-height: 40px;
}
QPushButton#accentBtn:hover {
    background-color: #2260aa;
}
QPushButton#accentBtn:disabled {
    background-color: #0d2244;
    color: #335577;
    border-color: #1a3355;
}
QPushButton#dangerBtn {
    background-color: #4a1a1a;
    border-color: #883333;
    color: #ffaaaa;
}
QPushButton#dangerBtn:hover {
    background-color: #662222;
}
QDoubleSpinBox, QSpinBox, QComboBox {
    background-color: #151530;
    border: 1px solid #2a2a5a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #c0c0e0;
    font-size: 12px;
    min-height: 24px;
}
QProgressBar {
    border: 1px solid #2a2a5a;
    border-radius: 4px;
    background-color: #0a0a1a;
    text-align: center;
    color: #8899cc;
    font-size: 11px;
}
QProgressBar::chunk {
    background-color: #2a5599;
    border-radius: 3px;
}
QLabel {
    color: #a0a0c0;
    font-size: 12px;
}
QLabel#headerLabel {
    color: #6699dd;
    font-size: 15px;
    font-weight: bold;
}
QLabel#valueLabel {
    color: #88bbee;
    font-size: 13px;
    font-weight: bold;
}
QRadioButton {
    color: #a0a0c0;
    font-size: 12px;
    spacing: 6px;
}
QRadioButton::indicator {
    width: 14px;
    height: 14px;
}
QStatusBar {
    background-color: #0a0a15;
    color: #667799;
    font-size: 11px;
    border-top: 1px solid #1a1a3a;
}
QSplitter::handle {
    background-color: #1a1a3a;
    width: 2px;
}
"""


class DemoWindow(QMainWindow):
    def __init__(self, zaber_port: str = None, xpr_port: str = None):
        super().__init__()
        self.setWindowTitle("ENPH 459 — Super-Resolution Demo")
        self.setMinimumSize(1400, 850)
        self.setStyleSheet(DARK_STYLE)

        self._zaber_port = zaber_port
        self._xpr_port = xpr_port
        self._imaging_sys = None
        self._zaber_conn = None
        self._zaber_axis = None
        self._is_color = False
        self._worker = None
        self._live_feed = None

        # Captured data
        self._captured_images = []
        self._captured_shifts = []

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ── Left: Hardware Controls ──────────────────────────────────────
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)

        # Connection
        conn_group = QGroupBox("Hardware Connection")
        conn_lay = QVBoxLayout(conn_group)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Zaber:"))
        self.zaber_port_edit = QComboBox()
        self.zaber_port_edit.setEditable(True)
        self.zaber_port_edit.addItems(["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0"])
        if zaber_port:
            self.zaber_port_edit.setCurrentText(zaber_port)
        port_row.addWidget(self.zaber_port_edit)
        conn_lay.addLayout(port_row)

        port_row2 = QHBoxLayout()
        port_row2.addWidget(QLabel("XPR:"))
        self.xpr_port_edit = QComboBox()
        self.xpr_port_edit.setEditable(True)
        self.xpr_port_edit.addItems(["/dev/ttyACM1", "/dev/ttyACM0", "/dev/ttyUSB1"])
        if xpr_port:
            self.xpr_port_edit.setCurrentText(xpr_port)
        port_row2.addWidget(self.xpr_port_edit)
        conn_lay.addLayout(port_row2)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setObjectName("accentBtn")
        self.connect_btn.clicked.connect(self._connect_hardware)
        conn_lay.addWidget(self.connect_btn)

        self.conn_status = QLabel("Disconnected")
        self.conn_status.setAlignment(Qt.AlignCenter)
        conn_lay.addWidget(self.conn_status)

        self.autoexp_btn = QPushButton("Auto-Exposure")
        self.autoexp_btn.clicked.connect(self._run_auto_exposure)
        self.autoexp_btn.setEnabled(False)
        conn_lay.addWidget(self.autoexp_btn)

        left_layout.addWidget(conn_group)

        # Stage movement
        stage_group = QGroupBox("Stage Movement")
        stage_lay = QVBoxLayout(stage_group)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step (mm):"))
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.001, 10.0)
        self.step_spin.setValue(0.1)
        self.step_spin.setDecimals(3)
        self.step_spin.setSingleStep(0.05)
        step_row.addWidget(self.step_spin)
        stage_lay.addLayout(step_row)

        # X row
        x_row = QHBoxLayout()
        self.x_minus = QPushButton("X -")
        self.x_plus = QPushButton("X +")
        self.x_minus.clicked.connect(lambda: self._jog_stage("x", -1))
        self.x_plus.clicked.connect(lambda: self._jog_stage("x", +1))
        x_row.addWidget(self.x_minus)
        x_row.addWidget(self.x_plus)
        stage_lay.addLayout(x_row)

        # Y row
        y_row = QHBoxLayout()
        self.y_minus = QPushButton("Y -")
        self.y_plus = QPushButton("Y +")
        self.y_minus.clicked.connect(lambda: self._jog_stage("y", -1))
        self.y_plus.clicked.connect(lambda: self._jog_stage("y", +1))
        y_row.addWidget(self.y_minus)
        y_row.addWidget(self.y_plus)
        stage_lay.addLayout(y_row)

        # Z row
        z_row = QHBoxLayout()
        self.z_minus = QPushButton("Z -")
        self.z_plus = QPushButton("Z +")
        self.z_minus.clicked.connect(lambda: self._jog_stage("z", -1))
        self.z_plus.clicked.connect(lambda: self._jog_stage("z", +1))
        z_row.addWidget(self.z_minus)
        z_row.addWidget(self.z_plus)
        stage_lay.addLayout(z_row)

        self.stage_pos_label = QLabel("Position: --")
        self.stage_pos_label.setObjectName("valueLabel")
        self.stage_pos_label.setAlignment(Qt.AlignCenter)
        stage_lay.addWidget(self.stage_pos_label)

        left_layout.addWidget(stage_group)

        # Autofocus
        af_group = QGroupBox("Autofocus (X axis)")
        af_lay = QVBoxLayout(af_group)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Range (mm):"))
        self.af_range_spin = QDoubleSpinBox()
        self.af_range_spin.setRange(0.1, 100.0)
        self.af_range_spin.setValue(25.0)
        self.af_range_spin.setDecimals(1)
        self.af_range_spin.setSingleStep(1.0)
        range_row.addWidget(self.af_range_spin)
        af_lay.addLayout(range_row)

        step_af_row = QHBoxLayout()
        step_af_row.addWidget(QLabel("Step (mm):"))
        self.af_step_spin = QDoubleSpinBox()
        self.af_step_spin.setRange(0.01, 10.0)
        self.af_step_spin.setValue(1.0)
        self.af_step_spin.setDecimals(2)
        self.af_step_spin.setSingleStep(0.25)
        step_af_row.addWidget(self.af_step_spin)
        af_lay.addLayout(step_af_row)

        self.autofocus_btn = QPushButton("Run Autofocus")
        self.autofocus_btn.setObjectName("accentBtn")
        self.autofocus_btn.clicked.connect(self._run_autofocus)
        self.autofocus_btn.setEnabled(False)
        af_lay.addWidget(self.autofocus_btn)

        self.af_progress = QProgressBar()
        self.af_progress.setTextVisible(True)
        af_lay.addWidget(self.af_progress)

        left_layout.addWidget(af_group)
        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # ── Middle: Capture Settings ─────────────────────────────────────
        mid_panel = QWidget()
        mid_layout = QVBoxLayout(mid_panel)
        mid_layout.setSpacing(10)

        # Pixel shift mode
        mode_group = QGroupBox("Pixel Shift Mode")
        mode_lay = QVBoxLayout(mode_group)
        self.mode_group = QButtonGroup(self)
        self.half_px_radio = QRadioButton("Half pixel (0.5 px) — mono")
        self.full_px_radio = QRadioButton("Full pixel (1.0 px) — color/Bayer")
        self.half_px_radio.setChecked(True)
        self.mode_group.addButton(self.half_px_radio, 0)
        self.mode_group.addButton(self.full_px_radio, 1)
        mode_lay.addWidget(self.half_px_radio)
        mode_lay.addWidget(self.full_px_radio)
        mid_layout.addWidget(mode_group)

        # Pattern selection
        pattern_group = QGroupBox("Capture Pattern")
        pat_lay = QGridLayout(pattern_group)
        pat_lay.setSpacing(8)

        self._pattern_widgets = {}
        self._selected_pattern = "4 Corners"
        for i, (name, signs) in enumerate(SHIFT_PATTERNS.items()):
            pw = PatternPreview(name, signs)
            pw.clicked.connect(lambda n=name: self._select_pattern(n))
            pat_lay.addWidget(pw, 0, i, Qt.AlignCenter)
            self._pattern_widgets[name] = pw

        self._pattern_widgets["4 Corners"].selected = True
        mid_layout.addWidget(pattern_group)

        # Capture settings
        cap_group = QGroupBox("Capture Settings")
        cap_lay = QVBoxLayout(cap_group)

        settle_row = QHBoxLayout()
        settle_row.addWidget(QLabel("Settle (ms):"))
        self.settle_spin = QSpinBox()
        self.settle_spin.setRange(5, 500)
        self.settle_spin.setValue(50)
        settle_row.addWidget(self.settle_spin)
        cap_lay.addLayout(settle_row)

        mid_layout.addWidget(cap_group)

        # Capture button
        self.capture_btn = QPushButton("Start Capture")
        self.capture_btn.setObjectName("accentBtn")
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self._start_capture)
        mid_layout.addWidget(self.capture_btn)

        self.capture_progress = QProgressBar()
        self.capture_progress.setTextVisible(True)
        mid_layout.addWidget(self.capture_progress)

        # SR scope option
        scope_group = QGroupBox("Scope")
        scope_lay = QVBoxLayout(scope_group)
        self.sr_scope_group = QButtonGroup(self)
        self.roi_only_radio = QRadioButton("ROI only")
        self.full_then_crop_radio = QRadioButton("Full image")
        self.roi_only_radio.setChecked(True)
        self.sr_scope_group.addButton(self.roi_only_radio, 0)
        self.sr_scope_group.addButton(self.full_then_crop_radio, 1)
        scope_lay.addWidget(self.roi_only_radio)
        scope_lay.addWidget(self.full_then_crop_radio)
        mid_layout.addWidget(scope_group)

        psf_group = QGroupBox("PSF")
        psf_lay = QVBoxLayout(psf_group)
        self.psf_group = QButtonGroup(self)
        self.measured_psf_radio = QRadioButton("Measured")
        self.gaussian_psf_radio = QRadioButton("Gaussian")
        self.measured_psf_radio.setChecked(True)
        self.psf_group.addButton(self.measured_psf_radio, 0)
        self.psf_group.addButton(self.gaussian_psf_radio, 1)
        psf_lay.addWidget(self.measured_psf_radio)
        psf_lay.addWidget(self.gaussian_psf_radio)
        mid_layout.addWidget(psf_group)

        # Reconstruct button
        self.reconstruct_btn = QPushButton("Run Super-Resolution")
        self.reconstruct_btn.setObjectName("accentBtn")
        self.reconstruct_btn.setEnabled(False)
        self.reconstruct_btn.clicked.connect(self._run_reconstruction)
        mid_layout.addWidget(self.reconstruct_btn)

        self.recon_progress = QProgressBar()
        self.recon_progress.setTextVisible(True)
        mid_layout.addWidget(self.recon_progress)

        # Autofocus plot (fills remaining space)
        af_plot_group = QGroupBox("Autofocus Result")
        af_plot_lay = QVBoxLayout(af_plot_group)
        self.af_figure = Figure(figsize=(4, 3), facecolor='#0f0f1a')
        self.af_canvas = FigureCanvas(self.af_figure)
        self.af_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.af_canvas.setMinimumHeight(200)
        af_plot_lay.addWidget(self.af_canvas)
        mid_layout.addWidget(af_plot_group, stretch=1)

        splitter.addWidget(mid_panel)

        # ── Right: Live Preview + Results ────────────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)

        # Live preview (large) with ROI selection
        preview_group = QGroupBox("Live Preview  (drag to select ROI)")
        preview_lay = QVBoxLayout(preview_group)
        self.preview_label = ROIPreviewLabel()
        self.preview_label.setMinimumSize(500, 400)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.roi_changed.connect(self._on_roi_changed)
        preview_lay.addWidget(self.preview_label)
        right_layout.addWidget(preview_group)

        # SR results
        results_label = QLabel("Super-Resolution Results")
        results_label.setObjectName("headerLabel")
        results_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(results_label)

        self.results_figure = Figure(figsize=(7, 4), facecolor='#0f0f1a')
        self.results_canvas = FigureCanvas(self.results_figure)
        self.results_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.results_canvas)

        splitter.addWidget(right_panel)
        splitter.setSizes([300, 280, 800])

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — connect hardware to begin")

    # ── Hardware ─────────────────────────────────────────────────────────

    def _connect_hardware(self):
        if self._imaging_sys is not None:
            self._disconnect_hardware()
            return

        zaber_port = self.zaber_port_edit.currentText().strip()
        xpr_port = self.xpr_port_edit.currentText().strip()

        # Disable button while connecting
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("Connecting...")

        worker = ConnectWorker(xpr_port, zaber_port)
        worker.status_msg.connect(lambda msg: self.status.showMessage(msg))
        worker.finished.connect(self._on_connect_finished)
        worker.error.connect(self._on_connect_error)
        self._worker = worker
        worker.start()

    def _on_connect_finished(self, imaging_sys, zaber_conn, zaber_axis,
                             auto_exp, is_color):
        self._imaging_sys = imaging_sys
        self._zaber_conn = zaber_conn
        self._zaber_axis = zaber_axis
        self._is_color = is_color

        if is_color:
            self.full_px_radio.setChecked(True)
        else:
            self.half_px_radio.setChecked(True)

        if zaber_axis:
            try:
                from zaber_motion import Units
                pos = zaber_axis.get_position(Units.LENGTH_MILLIMETRES)
                self.stage_pos_label.setText(f"X: {pos:.3f} mm")
            except Exception:
                pass

        cam = self._imaging_sys.camera
        cam_type = "color" if is_color else "mono"
        self.conn_status.setText(
            f"Connected: {cam.width}x{cam.height} {cam_type}\n"
            f"Exp: {cam.exposure:.0f} us")
        self.conn_status.setStyleSheet("color: #66dd88;")
        self.connect_btn.setText("Disconnect")
        self.connect_btn.setEnabled(True)
        self.connect_btn.setObjectName("dangerBtn")
        self.connect_btn.setStyleSheet(self.styleSheet())

        self.autofocus_btn.setEnabled(zaber_axis is not None)
        self.capture_btn.setEnabled(True)
        self.autoexp_btn.setEnabled(True)
        self._set_stage_buttons_enabled(zaber_axis is not None)

        # Start live preview feed (~5 Hz)
        self._start_live_feed()

        self.status.showMessage(
            f"Connected — {cam_type} camera {cam.width}x{cam.height}, "
            f"exposure {cam.exposure:.0f} us")

    def _on_connect_error(self, msg):
        self._disconnect_hardware()
        self.connect_btn.setEnabled(True)
        first_line = msg.splitlines()[0] if msg else "Unknown error"
        QMessageBox.critical(self, "Connection Error", first_line)
        self.status.showMessage(f"Connection failed: {first_line}")

    def _start_live_feed(self):
        self._stop_live_feed()
        if self._imaging_sys:
            self._live_feed = LiveFeedWorker(self._imaging_sys, self._is_color)
            self._live_feed.frame.connect(self._show_preview)
            self._live_feed.start()

    def _stop_live_feed(self):
        if self._live_feed:
            self._live_feed.stop()
            self._live_feed.wait(2000)
            self._live_feed = None

    def _pause_live_feed(self):
        if self._live_feed:
            self._live_feed.pause()

    def _resume_live_feed(self):
        if self._live_feed:
            self._live_feed.resume()

    def _disconnect_hardware(self):
        self._stop_live_feed()
        if self._imaging_sys:
            try:
                self._imaging_sys.close()
            except Exception:
                pass
            self._imaging_sys = None

        if self._zaber_conn:
            try:
                self._zaber_conn.close()
            except Exception:
                pass
            self._zaber_conn = None
            self._zaber_axis = None

        self.conn_status.setText("Disconnected")
        self.conn_status.setStyleSheet("color: #aa6666;")
        self.connect_btn.setText("Connect")
        self.connect_btn.setEnabled(True)
        self.connect_btn.setObjectName("accentBtn")
        self.connect_btn.setStyleSheet(self.styleSheet())
        self.autofocus_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.reconstruct_btn.setEnabled(False)
        self.autoexp_btn.setEnabled(False)
        self._set_stage_buttons_enabled(False)
        self.stage_pos_label.setText("Position: --")
        self.status.showMessage("Disconnected")

    def _set_stage_buttons_enabled(self, enabled: bool):
        for btn in (self.x_minus, self.x_plus, self.y_minus, self.y_plus,
                    self.z_minus, self.z_plus):
            btn.setEnabled(enabled)

    def _jog_stage(self, axis: str, direction: int):
        if not self._zaber_axis:
            return
        step = self.step_spin.value() * direction
        # Map axis name to Zaber axis number.
        # Lockstep group 1 is used for X (focus). Individual axes 1,2,3
        # are available for Y/Z depending on hardware.
        axis_map = {"x": 1, "y": 2, "z": 3}
        try:
            from zaber_motion import Units
            if axis == "x":
                # X uses lockstep (or single axis) stored at connect time
                self._zaber_axis.move_relative(step, Units.LENGTH_MILLIMETRES,
                                                wait_until_idle=True)
                pos = self._zaber_axis.get_position(Units.LENGTH_MILLIMETRES)
                self.stage_pos_label.setText(f"X: {pos:.3f} mm")
                self.status.showMessage(f"Moved X by {step:+.3f} mm → {pos:.3f} mm")
            else:
                # Try the axis on the already-detected device
                device = self._zaber_conn.detect_devices()[0]
                num_axes = device.axis_count
                ax_num = axis_map[axis]
                if ax_num > num_axes:
                    self.status.showMessage(
                        f"{axis.upper()} axis #{ax_num} not available "
                        f"(device has {num_axes} axes)")
                    return
                ax = device.get_axis(ax_num)
                ax.move_relative(step, Units.LENGTH_MILLIMETRES,
                                 wait_until_idle=True)
                pos = ax.get_position(Units.LENGTH_MILLIMETRES)
                self.status.showMessage(
                    f"Moved {axis.upper()} by {step:+.3f} mm → {pos:.3f} mm")
        except Exception as e:
            self.status.showMessage(f"Stage {axis.upper()} error: {e}")

    # ── Auto-exposure ──────────────────────────────────────────────────

    def _run_auto_exposure(self):
        if not self._imaging_sys:
            return
        self._pause_live_feed()
        self.autoexp_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        worker = AutoExposureWorker(self._imaging_sys, self._is_color)
        worker.status_msg.connect(lambda msg: self.status.showMessage(msg))
        worker.preview.connect(self._show_preview)
        worker.finished.connect(self._autoexp_on_finished)
        worker.error.connect(self._autoexp_on_error)
        self._worker = worker
        worker.start()

    def _autoexp_on_finished(self, exposure_us):
        self.autoexp_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)
        self._resume_live_feed()
        cam = self._imaging_sys.camera
        cam_type = "color" if self._is_color else "mono"
        self.conn_status.setText(
            f"Connected: {cam.width}x{cam.height} {cam_type}\n"
            f"Exp: {exposure_us:.0f} us")
        self.status.showMessage(f"Auto-exposure done — {exposure_us:.0f} us")

    def _autoexp_on_error(self, msg):
        self.autoexp_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)
        self._resume_live_feed()
        self.status.showMessage(f"Auto-exposure error: {msg.splitlines()[0]}")
        QMessageBox.warning(self, "Auto-Exposure Error", msg)

    # ── Autofocus ────────────────────────────────────────────────────────

    def _run_autofocus(self):
        if not self._zaber_axis or not self._imaging_sys:
            return
        self._pause_live_feed()

        from zaber_motion import Units
        start_pos = self._zaber_axis.get_position(Units.LENGTH_MILLIMETRES)

        self.autofocus_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.af_progress.setValue(0)

        self._af_positions = []
        self._af_variances = []

        worker = AutofocusWorker(
            self._zaber_axis, self._imaging_sys,
            self.af_range_spin.value(), self.af_step_spin.value(), start_pos,
            is_color=self._is_color)
        worker.progress.connect(self._af_on_progress)
        worker.preview.connect(self._show_preview)
        worker.finished.connect(self._af_on_finished)
        worker.error.connect(self._af_on_error)
        self._worker = worker
        worker.start()

    def _af_plot(self, mark_best=False):
        """Redraw the autofocus plot with current data."""
        self.af_figure.clear()
        ax = self.af_figure.add_subplot(111)
        ax.set_facecolor('#0f0f1a')
        ax.plot(self._af_positions, self._af_variances, '-o',
                color='#6cb4ee', markersize=4, linewidth=1.5)
        if mark_best and self._af_variances:
            best_idx = int(np.argmax(self._af_variances))
            ax.axvline(self._af_positions[best_idx], color='#ff6666',
                       linestyle='--', linewidth=1,
                       label=f'Best: {self._af_positions[best_idx]:.2f}mm')
            ax.scatter([self._af_positions[best_idx]],
                       [self._af_variances[best_idx]],
                       color='#ff6666', s=60, zorder=5)
            ax.legend(fontsize=8, facecolor='#0f0f1a', edgecolor='#2a2a4a',
                      labelcolor='#aabbcc')
        ax.set_xlabel("Position (mm)", color='#8899aa', fontsize=9)
        ax.set_ylabel("Laplacian Var.", color='#8899aa', fontsize=9)
        ax.set_title("Autofocus", color='#8899cc', fontsize=10)
        ax.tick_params(colors='#667788', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#2a2a4a')
        self.af_figure.tight_layout()
        self.af_canvas.draw()

    def _af_on_progress(self, step, total, pos, lap_var):
        self.af_progress.setMaximum(total)
        self.af_progress.setValue(step)
        self.af_progress.setFormat(f"{step}/{total}  lap={lap_var:.0f}")
        self.status.showMessage(f"Autofocus: pos={pos:.3f}mm, laplacian={lap_var:.0f}")
        self._af_positions.append(pos)
        self._af_variances.append(lap_var)
        self._af_plot(mark_best=False)

    def _af_on_finished(self, best_pos, positions, variances):
        self.autofocus_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)
        self._resume_live_feed()
        self.af_progress.setFormat(f"Best: {best_pos:.3f} mm")
        self.stage_pos_label.setText(f"X: {best_pos:.3f} mm")
        self.status.showMessage(f"Autofocus complete — best position: {best_pos:.3f} mm")
        self._af_positions = list(positions)
        self._af_variances = list(variances)
        self._af_plot(mark_best=True)

    def _af_on_error(self, msg):
        self.autofocus_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)
        self._resume_live_feed()
        self.status.showMessage(f"Autofocus error: {msg.splitlines()[0]}")
        QMessageBox.warning(self, "Autofocus Error", msg)

    # ── Pattern selection ────────────────────────────────────────────────

    def _select_pattern(self, name: str):
        self._selected_pattern = name
        for n, pw in self._pattern_widgets.items():
            pw.selected = (n == name)

    # ── Capture ──────────────────────────────────────────────────────────

    def _start_capture(self):
        if not self._imaging_sys:
            return
        self._pause_live_feed()

        # Build positions from pattern + mode
        pattern_signs = SHIFT_PATTERNS[self._selected_pattern]
        is_half = self.half_px_radio.isChecked()
        target_px = 0.5 if is_half else 1.0

        # Get tilt angles from calibration
        cal_path = str(CALIBRATION_CSV)
        if CALIBRATION_CSV.exists():
            tilt_x, tilt_y = interpolate_tilt_for_shift(cal_path, target_px)
        else:
            tilt_x = NOMINAL_TILT_MONO if is_half else NOMINAL_TILT_COLOR
            tilt_y = tilt_x

        positions = [(sx * tilt_x, sy * tilt_y) for sx, sy in pattern_signs]

        self.capture_btn.setEnabled(False)
        self.reconstruct_btn.setEnabled(False)
        self.capture_progress.setValue(0)
        self.capture_progress.setMaximum(len(positions))

        worker = CaptureWorker(
            self._imaging_sys, positions,
            self.settle_spin.value(), self._is_color)
        worker.progress.connect(self._cap_on_progress)
        worker.image_captured.connect(self._cap_on_image)
        worker.finished.connect(self._cap_on_finished)
        worker.error.connect(self._cap_on_error)
        self._worker = worker
        worker.start()

    def _cap_on_progress(self, msg):
        self.status.showMessage(msg)
        val = self.capture_progress.value()
        self.capture_progress.setValue(val + 1)

    def _cap_on_image(self, idx, img):
        self._show_preview(img)

    def _cap_on_finished(self, images, shifts_px):
        self._captured_images = images
        self._captured_shifts = shifts_px
        self.capture_btn.setEnabled(True)
        self.reconstruct_btn.setEnabled(True)
        self._resume_live_feed()
        self.capture_progress.setFormat(f"{len(images)} images captured")
        self.status.showMessage(
            f"Capture complete — {len(images)} images. "
            f"Click 'Run Super-Resolution' to reconstruct.")

    def _cap_on_error(self, msg):
        self.capture_btn.setEnabled(True)
        self._resume_live_feed()
        self.status.showMessage(f"Capture error: {msg.splitlines()[0]}")
        QMessageBox.warning(self, "Capture Error", msg)

    def _show_preview(self, img: np.ndarray):
        """Show a numpy image in the ROI preview label."""
        if img.ndim == 3 and img.shape[2] == 3:
            display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 2:
            display = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            display = img

        img_h, img_w = display.shape[:2]
        display = np.ascontiguousarray(display)
        self._preview_buf = display
        qimg = QImage(display.data, img_w, img_h, img_w * 3, QImage.Format_RGB888)
        self.preview_label.set_image(QPixmap.fromImage(qimg), img_w, img_h)

    def _on_roi_changed(self, x, y, w, h):
        self.status.showMessage(f"ROI set: ({x}, {y}) {w}x{h} px")

    # ── Reconstruction ───────────────────────────────────────────────────

    def _run_reconstruction(self):
        if not self._captured_images:
            return

        self.reconstruct_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.recon_progress.setValue(0)
        self._ibp_maes = []

        # Use ROI from preview drag, or fall back to centre crop
        roi = self.preview_label.roi_image_coords  # (x, y, w, h) or None
        roi_only = self.roi_only_radio.isChecked()
        use_measured = self.measured_psf_radio.isChecked()

        worker = ReconstructionWorker(
            self._captured_images, roi=roi, is_color=self._is_color,
            roi_only=roi_only, use_measured_psf=use_measured)

        worker.progress.connect(lambda msg: self.status.showMessage(msg))
        worker.ibp_step.connect(self._recon_ibp_step)
        worker.finished.connect(self._recon_on_finished)
        worker.error.connect(self._recon_on_error)
        self._worker = worker
        worker.start()

    def _recon_ibp_step(self, step, total, mse):
        self.recon_progress.setMaximum(total)
        self.recon_progress.setValue(step)
        self.recon_progress.setFormat(f"IBP: {step}/{total}  MSE={mse:.2f}")
        self._ibp_maes.append(mse)
        # Live MSE convergence plot in the autofocus plot area
        # Only redraw every 5 iterations to avoid slowdown
        if step % 5 == 0 or step == total:
            self.af_figure.clear()
            ax = self.af_figure.add_subplot(111)
            ax.set_facecolor('#0f0f1a')
            iters = list(range(1, len(self._ibp_maes) + 1))
            ax.plot(iters, self._ibp_maes, '-', color='#ee8866', linewidth=1.5)
            ax.set_xlabel("Iteration", color='#8899aa', fontsize=9)
            ax.set_ylabel("MSE", color='#8899aa', fontsize=9)
            ax.set_title("IBP Convergence", color='#ee9977', fontsize=10)
            ax.tick_params(colors='#667788', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#2a2a4a')
            self.af_figure.tight_layout()
            self.af_canvas.draw()

    def _recon_on_finished(self, original, saa, saa_ibp):
        self.reconstruct_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)
        self.recon_progress.setFormat("Done")
        self.status.showMessage("Super-resolution reconstruction complete")

        # Plot results
        self.results_figure.clear()

        # Bicubic 2x upsample of LR for comparison (matches "Native-2x")
        native_2x = ndi_zoom(original, UPSAMPLE_FACTOR, order=3)

        from matplotlib.colors import LinearSegmentedColormap
        red_cmap = LinearSegmentedColormap.from_list('red', ['#000000', '#ff0000'])

        imgs = [
            ("LR Red (bicubic 2x)", native_2x),
            ("SAA (Red)", saa),
            ("SAA + IBP (Red)", saa_ibp),
        ]

        for i, (title, img) in enumerate(imgs):
            ax = self.results_figure.add_subplot(1, 3, i + 1)
            ax.set_facecolor('#0a0a15')
            display = np.clip(img, 0, 255)
            ax.imshow(display, cmap=red_cmap, vmin=0, vmax=255,
                      interpolation='nearest')
            ax.set_title(title, color='#cc8899', fontsize=11, pad=8)
            ax.axis('off')

        self.results_figure.tight_layout(pad=1.0)
        self.results_canvas.draw()

    def _recon_on_error(self, msg):
        self.reconstruct_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)
        self.status.showMessage(f"Reconstruction error: {msg.splitlines()[0]}")
        QMessageBox.warning(self, "Reconstruction Error", msg)

    # ── Cleanup ──────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._disconnect_hardware()
        super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Super-Resolution Demo GUI")
    parser.add_argument("--zaber-port", default=None,
                        help="Zaber stage serial port (e.g. /dev/ttyACM0)")
    parser.add_argument("--xpr-port", default=None,
                        help="XPR controller serial port (e.g. /dev/ttyACM1)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DemoWindow(zaber_port=args.zaber_port, xpr_port=args.xpr_port)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
