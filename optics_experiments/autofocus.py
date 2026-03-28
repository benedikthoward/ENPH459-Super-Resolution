"""
Autofocus GUI: live camera viewfinder + Zaber stage jog + autofocus.

    uv run python -m optics_experiments.autofocus --port /dev/ttyUSB0
"""

import csv
import json
import sys
import argparse
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from PyQt5.QtCore import Qt, QThread, QRect, QPoint, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QSlider, QDoubleSpinBox, QSpinBox, QComboBox, QGroupBox,
    QCheckBox, QMessageBox, QRubberBand,
)

from zaber_motion import Units
from zaber_motion.ascii import Connection

from imaging.camera import DahengCamera
from optics_experiments.psf_mtf import (
    find_peak, extract_psf, fit_gaussian_psf, compute_mtf, mtf_at_fraction,
)


def laplacian_variance(gray: np.ndarray, roi=None) -> float:
    if roi is not None:
        x, y, w, h = roi
        gray = gray[y:y+h, x:x+w]
    if gray.size == 0:
        return 0.0
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def peak_intensity(gray: np.ndarray, roi=None) -> float:
    if roi is not None:
        x, y, w, h = roi
        gray = gray[y:y+h, x:x+w]
    if gray.size == 0:
        return 0.0
    return float(np.max(gray))


def encircled_energy_ratio(gray: np.ndarray, roi=None) -> float:
    """Fraction of total energy within 5px of the centroid — higher = tighter spot."""
    if roi is not None:
        x, y, w, h = roi
        gray = gray[y:y+h, x:x+w]
    if gray.size == 0:
        return 0.0
    img = gray.astype(np.float64)
    total = img.sum()
    if total == 0:
        return 0.0
    ys, xs = np.mgrid[:img.shape[0], :img.shape[1]]
    cx = (xs * img).sum() / total
    cy = (ys * img).sum() / total
    radius = 5
    dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
    core_energy = img[dist_sq <= radius ** 2].sum()
    return core_energy / total


def normalized_variance(gray: np.ndarray, roi=None) -> float:
    if roi is not None:
        x, y, w, h = roi
        gray = gray[y:y+h, x:x+w]
    if gray.size == 0:
        return 0.0
    img = gray.astype(np.float64)
    mean = img.mean()
    if mean == 0:
        return 0.0
    return img.var() / mean


def fwhm_metric(gray: np.ndarray, roi=None) -> float:
    """Inverse FWHM from 2D Gaussian PSF fit — higher = tighter spot = better focus."""
    if roi is not None:
        x, y, w, h = roi
        gray = gray[y:y+h, x:x+w]
    if gray.size == 0:
        return 0.0
    img = gray.astype(np.float64)
    peak = find_peak(img)
    radius = min(50, min(img.shape) // 2 - 1)
    if radius < 5:
        return 0.0
    psf = extract_psf(img, peak, radius)
    popt, _ = fit_gaussian_psf(psf)
    if popt is None:
        return 0.0
    sigma_x, sigma_y = popt[3], popt[4]
    fwhm = 2.3548 * (sigma_x + sigma_y) / 2.0  # average FWHM
    if fwhm < 0.1:
        return 0.0
    return 1.0 / fwhm  # invert so higher = better focus


FOCUS_METRICS = {
    "Laplacian Variance": laplacian_variance,
    "Peak Intensity": peak_intensity,
    "Encircled Energy": encircled_energy_ratio,
    "Normalized Variance": normalized_variance,
    "FWHM (point source)": fwhm_metric,
}

DEFAULT_METRIC = "Laplacian Variance"


def _get_limit(axis, setting: str, fallback: float) -> float:
    try:
        return axis.settings.get(setting, Units.LENGTH_MILLIMETRES)
    except Exception:
        return fallback


def _move_clamped(axis, delta_mm: float, lim_min: float, lim_max: float):
    current = axis.get_position(Units.LENGTH_MILLIMETRES)
    target = max(lim_min, min(lim_max, current + delta_mm))
    if abs(target - current) < 1e-4:
        return
    axis.move_absolute(target, Units.LENGTH_MILLIMETRES, wait_until_idle=True)


# ── Viewfinder ──────────────────────────────────────────────────────────────

class ViewfinderLabel(QLabel):
    roi_changed = pyqtSignal(object)  # emits (x, y, w, h) or None

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: black;")
        self.setCursor(Qt.CrossCursor)

        self._pixmap = None
        self._frame_shape = None  # (H, W)
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self._roi_img = None  # (x, y, w, h) in image coords
        self._focus_metric = 0.0

        self._rubber = QRubberBand(QRubberBand.Rectangle, self)
        self._drag_origin = None

    def set_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        self._frame_shape = (h, w)
        if frame.ndim == 2:
            qimg = QImage(frame.data, w, h, w, QImage.Format_Grayscale8)
        else:
            qimg = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg)
        scaled = pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self._pixmap = scaled
        self._scale = scaled.width() / w
        self._offset = QPoint(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
        )
        self.update()

    def set_focus_metric(self, val: float):
        self._focus_metric = val
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        if self._pixmap:
            p.drawPixmap(self._offset, self._pixmap)

        # draw ROI
        if self._roi_img and self._frame_shape:
            x, y, w, h = self._roi_img
            rx = int(x * self._scale) + self._offset.x()
            ry = int(y * self._scale) + self._offset.y()
            rw = int(w * self._scale)
            rh = int(h * self._scale)
            p.setPen(QPen(QColor(0, 255, 0), 2))
            p.drawRect(rx, ry, rw, rh)

        # draw focus metric
        p.setPen(QPen(QColor(0, 255, 0)))
        p.setFont(QFont("Monospace", 14, QFont.Bold))
        p.drawText(self._offset + QPoint(10, 25), f"Focus: {self._focus_metric:.4g}")
        p.end()

    def _widget_to_img(self, pos: QPoint):
        x = (pos.x() - self._offset.x()) / self._scale
        y = (pos.y() - self._offset.y()) / self._scale
        if self._frame_shape:
            h, w = self._frame_shape
            x = max(0, min(w, x))
            y = max(0, min(h, y))
        return x, y

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._frame_shape:
            self._drag_origin = event.pos()
            self._rubber.setGeometry(QRect(event.pos(), event.pos()))
            self._rubber.show()

    def mouseMoveEvent(self, event):
        if self._drag_origin:
            self._rubber.setGeometry(QRect(self._drag_origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if self._drag_origin and self._frame_shape:
            self._rubber.hide()
            x1, y1 = self._widget_to_img(self._drag_origin)
            x2, y2 = self._widget_to_img(event.pos())
            ix, iy = int(min(x1, x2)), int(min(y1, y2))
            iw, ih = int(abs(x2 - x1)), int(abs(y2 - y1))
            if iw > 10 and ih > 10:
                self._roi_img = (ix, iy, iw, ih)
            else:
                self._roi_img = None
            self.roi_changed.emit(self._roi_img)
            self._drag_origin = None
            self.update()


# ── Camera thread ───────────────────────────────────────────────────────────

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    focus_metric_ready = pyqtSignal(float)

    def __init__(self, cam: DahengCamera):
        super().__init__()
        self.cam = cam
        self.running = True
        self.roi = None
        self.metric_fn = laplacian_variance

    def run(self):
        while self.running:
            try:
                if self.cam.is_color:
                    frame = self.cam.capture_rgb()
                else:
                    frame = self.cam.capture_raw()
            except Exception:
                self.msleep(100)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
            metric = self.metric_fn(gray, self.roi)
            self.frame_ready.emit(frame)
            self.focus_metric_ready.emit(metric)


# ── Autofocus worker ────────────────────────────────────────────────────────

def _compute_mtf50(frame: np.ndarray, roi=None, psf_mode: bool = False) -> float:
    """Compute MTF50 (cycles/pixel) from a frame.

    If psf_mode is True, treats the (ROI of the) image as a direct PSF
    and computes the MTF from it via FFT. Otherwise, extracts the PSF
    from the brightest spot first.
    """
    if roi is not None:
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]
    if frame.size == 0:
        return 0.0
    img = frame.astype(np.float64)

    if psf_mode:
        # image IS the PSF — subtract background and compute MTF directly
        bg = np.median(img)
        psf = np.clip(img - bg, 0, None)
    else:
        peak = find_peak(img)
        radius = min(50, min(img.shape) // 2 - 1)
        if radius < 5:
            return 0.0
        psf = extract_psf(img, peak, radius)

    freq, mtf_profile, _, _, _ = compute_mtf(psf)
    mtf50 = mtf_at_fraction(freq, mtf_profile, 0.5)
    return float(mtf50) if not np.isnan(mtf50) else 0.0


class AutofocusWorker(QThread):
    progress = pyqtSignal(str)
    # positions, metrics, mtf50s, best_pos
    finished = pyqtSignal(list, list, list, float)

    def __init__(self, cam, axis, af_min, af_max, coarse_steps, fine_steps, roi, metric_fn,
                 save_images_dir=None, psf_mode=False):
        super().__init__()
        self.cam = cam
        self.axis = axis
        self.af_min = af_min
        self.af_max = af_max
        self.coarse_steps = coarse_steps
        self.fine_steps = fine_steps
        self.roi = roi
        self.metric_fn = metric_fn
        self.save_images_dir = save_images_dir
        self.psf_mode = psf_mode

    def run(self):
        positions = []
        metrics = []
        mtf50s = []

        # coarse sweep
        coarse_pos = np.linspace(self.af_min, self.af_max, self.coarse_steps)
        for i, pos in enumerate(coarse_pos):
            self.progress.emit(f"Coarse {i+1}/{self.coarse_steps}: {pos:.3f} mm")
            self.axis.move_absolute(pos, Units.LENGTH_MILLIMETRES, wait_until_idle=True)
            time.sleep(1.0)  # 1s mechanical settle
            if self.cam.is_color:
                frame = self.cam.capture_rgb()
            else:
                frame = self.cam.capture_raw()
            if self.save_images_dir:
                save_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 else frame
                cv2.imwrite(str(self.save_images_dir / f"coarse_{i:03d}_{pos:.3f}mm.png"), save_frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
            m = self.metric_fn(gray, self.roi)
            m50 = _compute_mtf50(gray, self.roi, psf_mode=self.psf_mode)
            positions.append(pos)
            metrics.append(m)
            mtf50s.append(m50)

        # find coarse peak
        best_idx = int(np.argmax(metrics))
        coarse_step = coarse_pos[1] - coarse_pos[0] if len(coarse_pos) > 1 else 1.0
        fine_min = max(self.af_min, coarse_pos[best_idx] - coarse_step)
        fine_max = min(self.af_max, coarse_pos[best_idx] + coarse_step)

        # fine sweep
        fine_pos = np.linspace(fine_min, fine_max, self.fine_steps)
        for i, pos in enumerate(fine_pos):
            self.progress.emit(f"Fine {i+1}/{self.fine_steps}: {pos:.3f} mm")
            self.axis.move_absolute(pos, Units.LENGTH_MILLIMETRES, wait_until_idle=True)
            time.sleep(1.0)  # 1s mechanical settle
            if self.cam.is_color:
                frame = self.cam.capture_rgb()
            else:
                frame = self.cam.capture_raw()
            if self.save_images_dir:
                save_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 else frame
                cv2.imwrite(str(self.save_images_dir / f"fine_{i:03d}_{pos:.3f}mm.png"), save_frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
            m = self.metric_fn(gray, self.roi)
            m50 = _compute_mtf50(gray, self.roi, psf_mode=self.psf_mode)
            positions.append(pos)
            metrics.append(m)
            mtf50s.append(m50)

        # move to best position overall
        best_overall = int(np.argmax(metrics))
        best_pos = positions[best_overall]
        self.axis.move_absolute(best_pos, Units.LENGTH_MILLIMETRES, wait_until_idle=True)
        self.progress.emit(f"Best focus: {best_pos:.3f} mm")
        self.finished.emit(positions, metrics, mtf50s, best_pos)


# ── Main GUI ────────────────────────────────────────────────────────────────

class AutofocusGUI(QMainWindow):
    def __init__(self, port: str):
        super().__init__()
        self.setWindowTitle("Autofocus")
        self._cam = None
        self._conn = None
        self._axes = {}  # {"X": axis, "Y": axis, "Z": axis}
        self._limits = {}  # {"X": (min, max), ...}
        self._cam_thread = None
        self._af_worker = None
        self._roi = None

        self._build_ui()
        self._connect_hardware(port)
        self._start_live()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # viewfinder
        self._viewfinder = ViewfinderLabel()
        self._viewfinder.roi_changed.connect(self._on_roi_changed)
        layout.addWidget(self._viewfinder, stretch=3)

        # controls panel
        controls = QVBoxLayout()
        controls.setSpacing(6)
        layout.addLayout(controls, stretch=0)

        # ── camera ──
        cam_box = QGroupBox("Camera")
        cam_lay = QVBoxLayout()

        cam_lay.addWidget(QLabel("Exposure (µs)"))
        self._exp_slider = QSlider(Qt.Horizontal)
        self._exp_slider.setRange(100, 500000)
        self._exp_slider.setValue(10000)
        self._exp_slider.valueChanged.connect(self._set_exposure)
        self._exp_label = QLabel("10000")
        cam_lay.addWidget(self._exp_slider)
        cam_lay.addWidget(self._exp_label)

        cam_lay.addWidget(QLabel("Gain (dB)"))
        self._gain_slider = QSlider(Qt.Horizontal)
        self._gain_slider.setRange(0, 240)  # 0-24 dB in 0.1 steps
        self._gain_slider.setValue(0)
        self._gain_slider.valueChanged.connect(self._set_gain)
        self._gain_label = QLabel("0.0")
        cam_lay.addWidget(self._gain_slider)
        cam_lay.addWidget(self._gain_label)

        cam_box.setLayout(cam_lay)
        controls.addWidget(cam_box)

        # ── stage jog ──
        jog_box = QGroupBox("Stage")
        jog_lay = QVBoxLayout()

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step (mm):"))
        self._step_spin = QDoubleSpinBox()
        self._step_spin.setRange(0.001, 10.0)
        self._step_spin.setValue(0.1)
        self._step_spin.setDecimals(3)
        self._step_spin.setSingleStep(0.01)
        step_row.addWidget(self._step_spin)
        jog_lay.addLayout(step_row)

        self._pos_labels = {}
        for axis_name in ["X", "Y", "Z"]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{axis_name}:"))
            btn_minus = QPushButton("<")
            btn_home = QPushButton("H")
            btn_plus = QPushButton(">")
            btn_minus.setFixedWidth(40)
            btn_home.setFixedWidth(40)
            btn_plus.setFixedWidth(40)
            btn_minus.clicked.connect(lambda _, a=axis_name: self._jog(a, -1))
            btn_home.clicked.connect(lambda _, a=axis_name: self._home_axis(a))
            btn_plus.clicked.connect(lambda _, a=axis_name: self._jog(a, +1))
            pos_label = QLabel("-- mm")
            pos_label.setFixedWidth(80)
            self._pos_labels[axis_name] = pos_label
            row.addWidget(btn_minus)
            row.addWidget(btn_home)
            row.addWidget(btn_plus)
            row.addWidget(pos_label)
            jog_lay.addLayout(row)

        jog_box.setLayout(jog_lay)
        controls.addWidget(jog_box)

        # ── autofocus ──
        af_box = QGroupBox("Autofocus")
        af_lay = QVBoxLayout()

        af_lay.addWidget(QLabel("Focus Axis:"))
        self._axis_combo = QComboBox()
        self._axis_combo.addItems(["X", "Y", "Z"])
        self._axis_combo.setCurrentText("Z")
        self._axis_combo.currentTextChanged.connect(self._on_focus_axis_changed)
        af_lay.addWidget(self._axis_combo)

        af_lay.addWidget(QLabel("Focus Metric:"))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems(list(FOCUS_METRICS.keys()))
        self._metric_combo.setCurrentText(DEFAULT_METRIC)
        self._metric_combo.currentTextChanged.connect(self._on_metric_changed)
        af_lay.addWidget(self._metric_combo)

        self._af_range_label = QLabel("Search range (mm):")
        af_lay.addWidget(self._af_range_label)

        af_lay.addWidget(QLabel("Min:"))
        self._af_min = QDoubleSpinBox()
        self._af_min.setRange(0, 500)
        self._af_min.setDecimals(2)
        self._af_min.setValue(0)
        af_lay.addWidget(self._af_min)

        af_lay.addWidget(QLabel("Max:"))
        self._af_max = QDoubleSpinBox()
        self._af_max.setRange(0, 500)
        self._af_max.setDecimals(2)
        self._af_max.setValue(50)
        af_lay.addWidget(self._af_max)

        self._af_half_range = QDoubleSpinBox()
        self._af_half_range.setRange(0.1, 50.0)
        self._af_half_range.setDecimals(1)
        self._af_half_range.setValue(5.0)
        self._af_half_range.setSuffix(" mm")
        half_row = QHBoxLayout()
        half_row.addWidget(QLabel("±range:"))
        half_row.addWidget(self._af_half_range)
        af_lay.addLayout(half_row)

        af_lay.addWidget(QLabel("Coarse steps:"))
        self._coarse_spin = QSpinBox()
        self._coarse_spin.setRange(5, 100)
        self._coarse_spin.setValue(20)
        af_lay.addWidget(self._coarse_spin)

        af_lay.addWidget(QLabel("Fine steps:"))
        self._fine_spin = QSpinBox()
        self._fine_spin.setRange(5, 50)
        self._fine_spin.setValue(10)
        af_lay.addWidget(self._fine_spin)

        self._psf_mode_cb = QCheckBox("Image is PSF (point source)")
        af_lay.addWidget(self._psf_mode_cb)

        self._save_graph_cb = QCheckBox("Save graph data")
        af_lay.addWidget(self._save_graph_cb)

        self._save_images_cb = QCheckBox("Save images")
        af_lay.addWidget(self._save_images_cb)

        self._af_btn = QPushButton("Autofocus")
        self._af_btn.clicked.connect(self._start_autofocus)
        af_lay.addWidget(self._af_btn)

        self._af_status = QLabel("Ready")
        self._af_status.setWordWrap(True)
        af_lay.addWidget(self._af_status)

        af_box.setLayout(af_lay)
        controls.addWidget(af_box)

        controls.addStretch()

    def _connect_hardware(self, port: str):
        # camera
        try:
            self._cam = DahengCamera()
            self._cam.exposure = self._exp_slider.value()
            self._cam.gain = self._gain_slider.value() / 10.0
        except Exception as e:
            QMessageBox.critical(self, "Camera Error", str(e))

        # zaber
        try:
            self._conn = Connection.open_serial_port(port)
            self._conn.enable_alerts()
            devices = self._conn.detect_devices()
            if not devices:
                raise RuntimeError("No Zaber devices found")
            device = devices[0]

            try:
                x_axis = device.get_lockstep(1)
                x_phys = device.get_axis(1)
            except Exception:
                x_axis = device.get_axis(1)
                x_phys = x_axis

            y_axis = device.get_axis(3)
            z_axis = device.get_axis(4)

            self._axes = {"X": x_axis, "Y": y_axis, "Z": z_axis}
            self._phys = {"X": x_phys, "Y": y_axis, "Z": z_axis}
            self._limits = {}
            for name, phys in self._phys.items():
                lo = _get_limit(phys, "limit.min", 0.0)
                hi = _get_limit(phys, "limit.max", 100.0)
                self._limits[name] = (lo, hi)

            self._update_pos_labels()
            self._on_focus_axis_changed(self._axis_combo.currentText())

        except Exception as e:
            QMessageBox.warning(self, "Zaber Error", str(e))
            self._af_btn.setEnabled(False)

    def _start_live(self):
        if self._cam is None:
            return
        self._cam_thread = CameraThread(self._cam)
        self._cam_thread.metric_fn = FOCUS_METRICS[self._metric_combo.currentText()]
        self._cam_thread.frame_ready.connect(self._on_frame)
        self._cam_thread.focus_metric_ready.connect(self._viewfinder.set_focus_metric)
        self._cam_thread.start()

    def _stop_live(self):
        if self._cam_thread:
            self._cam_thread.running = False
            self._cam_thread.wait()
            self._cam_thread = None

    def _on_frame(self, frame: np.ndarray):
        self._viewfinder.set_frame(frame)

    def _on_metric_changed(self, name: str):
        if self._cam_thread:
            self._cam_thread.metric_fn = FOCUS_METRICS[name]

    def _on_roi_changed(self, roi):
        self._roi = roi
        if self._cam_thread:
            self._cam_thread.roi = roi

    def _set_exposure(self, value):
        self._exp_label.setText(str(value))
        if self._cam:
            self._cam.exposure = value

    def _set_gain(self, value):
        gain_db = value / 10.0
        self._gain_label.setText(f"{gain_db:.1f}")
        if self._cam:
            self._cam.gain = gain_db

    def _update_pos_labels(self):
        for name, axis in self._axes.items():
            try:
                pos = axis.get_position(Units.LENGTH_MILLIMETRES)
                self._pos_labels[name].setText(f"{pos:.3f} mm")
            except Exception:
                self._pos_labels[name].setText("-- mm")

    def _update_af_range_from_current(self, axis_name: str):
        """Auto-populate AF min/max as ±half_range around current position."""
        if axis_name != self._axis_combo.currentText():
            return
        if axis_name not in self._axes:
            return
        try:
            pos = self._axes[axis_name].get_position(Units.LENGTH_MILLIMETRES)
        except Exception:
            return
        half = self._af_half_range.value()
        lo, hi = self._limits.get(axis_name, (0, 100))
        self._af_min.setValue(max(lo, pos - half))
        self._af_max.setValue(min(hi, pos + half))

    def _on_focus_axis_changed(self, axis_name: str):
        lo, hi = self._limits.get(axis_name, (0, 100))
        self._af_range_label.setText(f"Search range (mm) — {axis_name} limits: [{lo:.1f}, {hi:.1f}]")
        self._update_af_range_from_current(axis_name)

    def _jog(self, axis_name: str, direction: int):
        if axis_name not in self._axes:
            return
        step = self._step_spin.value() * direction
        lim_min, lim_max = self._limits[axis_name]
        try:
            _move_clamped(self._axes[axis_name], step, lim_min, lim_max)
            self._update_pos_labels()
            self._update_af_range_from_current(axis_name)
        except Exception as e:
            self._af_status.setText(f"Jog error: {e}")

    def _home_axis(self, axis_name: str):
        if axis_name not in self._axes:
            return
        self._af_status.setText(f"Homing {axis_name}...")
        QApplication.processEvents()
        try:
            self._axes[axis_name].home(wait_until_idle=True)
            self._af_status.setText(f"{axis_name} homed")
            self._update_pos_labels()
            self._update_af_range_from_current(axis_name)
        except Exception as e:
            self._af_status.setText(f"Home error: {e}")

    def _start_autofocus(self):
        focus_axis_name = self._axis_combo.currentText()
        if focus_axis_name not in self._axes:
            self._af_status.setText("No axis available")
            return

        self._af_btn.setEnabled(False)
        self._metric_combo.setEnabled(False)
        self._stop_live()

        # create save directory if saving images
        save_dir = None
        if self._save_images_cb.isChecked():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path("./autofocus_data") / ts
            save_dir.mkdir(parents=True, exist_ok=True)
            self._af_status.setText(f"Saving images to {save_dir}")

        self._save_dir = save_dir  # store for graph data saving

        self._af_worker = AutofocusWorker(
            cam=self._cam,
            axis=self._axes[focus_axis_name],
            af_min=self._af_min.value(),
            af_max=self._af_max.value(),
            coarse_steps=self._coarse_spin.value(),
            fine_steps=self._fine_spin.value(),
            roi=self._roi,
            metric_fn=FOCUS_METRICS[self._metric_combo.currentText()],
            save_images_dir=save_dir,
            psf_mode=self._psf_mode_cb.isChecked(),
        )
        self._af_worker.progress.connect(self._af_status.setText)
        self._af_worker.finished.connect(self._on_autofocus_done)
        self._af_worker.start()

    def _on_autofocus_done(self, positions, metrics, mtf50s, best_pos):
        self._af_btn.setEnabled(True)
        self._metric_combo.setEnabled(True)
        self._af_status.setText(f"Best focus: {best_pos:.3f} mm")
        self._start_live()
        self._show_focus_curve(positions, metrics, mtf50s, best_pos)

        # save graph data if checkbox is checked
        if self._save_graph_cb.isChecked():
            save_dir = self._save_dir
            if save_dir is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = Path("./autofocus_data") / ts
                save_dir.mkdir(parents=True, exist_ok=True)

            metric_name = self._metric_combo.currentText()

            # CSV
            csv_path = save_dir / "autofocus_data.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["position_mm", metric_name, "mtf50_cycles_per_px"])
                for pos, m, m50 in zip(positions, metrics, mtf50s):
                    writer.writerow([f"{pos:.4f}", f"{m:.6f}", f"{m50:.6f}"])

            # JSON
            json_path = save_dir / "autofocus_data.json"
            data = {
                "best_pos_mm": best_pos,
                "metric": metric_name,
                "positions_mm": positions,
                "metric_values": metrics,
                "mtf50_values": mtf50s,
            }
            json_path.write_text(json.dumps(data, indent=2))

            self._af_status.setText(f"Best: {best_pos:.3f} mm — data saved to {save_dir}")

    def _show_focus_curve(self, positions, metrics, mtf50s, best_pos):
        metric_name = self._metric_combo.currentText()
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # focus metric on left y-axis
        color1 = "tab:blue"
        ax1.plot(positions, metrics, "o-", markersize=4, color=color1, label=metric_name)
        ax1.axvline(best_pos, color="r", linestyle="--", label=f"Best: {best_pos:.3f} mm")
        ax1.set_xlabel("Position (mm)")
        ax1.set_ylabel(metric_name, color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        # MTF50 on right y-axis
        ax2 = ax1.twinx()
        color2 = "tab:orange"
        ax2.plot(positions, mtf50s, "s-", markersize=3, color=color2, alpha=0.7, label="MTF50")
        ax2.set_ylabel("MTF50 (cycles/pixel)", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        # combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        ax1.set_title(f"Autofocus Curve ({metric_name} + MTF50)")
        plt.tight_layout()
        plt.show(block=False)

    def closeEvent(self, event):
        self._stop_live()
        if self._cam:
            self._cam.close()
        if self._conn:
            self._conn.close()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Autofocus GUI")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Zaber serial port")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = AutofocusGUI(port=args.port)
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
