#!/usr/bin/env python3
"""
psf_sr_eval.py
==============
Super-resolution evaluation using PSF/pinhole images from the
20260325_184308 sweep calibration dataset.

Shift geometry
--------------
The calibration sweep moves the mirror along x and y axes independently.
Corner positions in the 3×3 grid are:  pos0(-x,+y)  pos2(+x,+y)
                                        pos6(-x,-y)  pos8(+x,-y)
  - x-sweep corners: only have ±x shift (dy ≈ 0)
  - y-sweep corners: only have ±y shift (dx ≈ 0)

For a full 2-D SR run we therefore select:
  1.  x-sweep pos0  →  shift (dy≈0, dx≈-0.5)
  2.  x-sweep pos2  →  shift (dy≈0, dx≈+0.5)
  3.  y-sweep pos0  →  shift (dy≈+0.5, dx≈0)
  4.  y-sweep pos6  →  shift (dy≈-0.5, dx≈0)
plus the center reference  x-sweep pos4  (no shift).

Two shift modes are compared
  - nominal : ±0.5 px (axis-aligned)
  - actual  : measured per-image offsets from centers.csv

Methods evaluated
-----------------
  Native-LR  : bicubic upscale of the center frame (baseline)
  SAA-nom    : Shift-and-Add with nominal shifts
  SAA-act    : Shift-and-Add with actual shifts
  IBP-nom    : SAA-nom + Iterative Back-Projection
  IBP-act    : SAA-act + Iterative Back-Projection

Evaluation
----------
Each output is cropped around the PSF, background-subtracted, and evaluated:
  - Gaussian fit → sigma_x, sigma_y, FWHM
  - Radial MTF (FFT of PSF) → MTF50

Usage
-----
    python psf_sr_eval.py imaging/calibration/20260325_184308
        [--pixel-pitch-um 3.45]
        [--upsample 2]
        [--shift-min-px 0.35] [--shift-max-px 0.65]
        [--crop-radius 60]
        [--psf-halfwidth 4]
        [--ibp-iterations 20] [--ibp-step 0.5]
        [--output-dir DIR]
"""

import argparse
import csv
import glob
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.ndimage import (center_of_mass, gaussian_filter,
                           shift as ndi_shift, zoom as ndi_zoom)
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIXEL_PITCH_UM = 3.45   # sensor pixel pitch in µm

# SR corner images: (sweep_axis, position_id, nominal_shift_yx_lr)
# nominal_shift_yx_lr is in LR pixel units (positive dy = image shifted DOWN)
SR_CORNERS = [
    ('x', 0, ( 0.0, -0.5)),   # x-sweep pos0 → left  shift
    ('x', 2, ( 0.0, +0.5)),   # x-sweep pos2 → right shift
    ('y', 0, (+0.5,  0.0)),   # y-sweep pos0 → down  shift
    ('y', 6, (-0.5,  0.0)),   # y-sweep pos6 → up    shift
]
SR_CENTER = ('x', 4)           # reference: x-sweep pos4

METHODS = ['Native-LR', 'SAA-nom', 'SAA-act', 'IBP-nom', 'IBP-act']
COLORS  = {'Native-LR': 'C0', 'SAA-nom': 'C2', 'SAA-act': 'C1',
           'IBP-nom': 'C3', 'IBP-act': 'C4'}

# ---------------------------------------------------------------------------
# Image helpers  (shared with psf_mtf.py)
# ---------------------------------------------------------------------------

def load_gray(path):
    img = np.array(Image.open(path), dtype=np.float64)
    return img.mean(axis=2) if img.ndim == 3 else img


def find_peak(img, smooth_sigma=2.0):
    smoothed = gaussian_filter(img, sigma=smooth_sigma)
    return np.unravel_index(smoothed.argmax(), smoothed.shape)


def extract_psf(img, center, radius, bg_percentile=50.0):
    r, c = center
    h, w = img.shape
    r0, r1 = max(r - radius, 0), min(r + radius + 1, h)
    c0, c1 = max(c - radius, 0), min(c + radius + 1, w)
    roi = img[r0:r1, c0:c1].copy()
    cy, cx = roi.shape[0] // 2, roi.shape[1] // 2
    Y, X = np.ogrid[:roi.shape[0], :roi.shape[1]]
    inner = ((Y - cy)**2 + (X - cx)**2) < (radius * 0.6)**2
    mask = np.ones_like(roi, dtype=bool)
    mask[inner] = False
    bg = np.percentile(roi[mask], bg_percentile)
    roi -= bg
    roi[roi < 0] = 0
    return roi


def subpixel_centre(psf):
    thresh = psf.max() * 0.1
    masked = np.where(psf > thresh, psf, 0.0)
    return center_of_mass(masked)


def align_psf(psf, target_c):
    com = np.array(subpixel_centre(psf))
    s = np.array(target_c) - com
    return np.clip(ndi_shift(psf, s, order=3, mode='constant'), 0, None)


def radial_average(data_2d, center=None, max_radius=None):
    h, w = data_2d.shape
    cy, cx = (h / 2.0, w / 2.0) if center is None else center
    Y, X = np.mgrid[:h, :w]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    if max_radius is None:
        max_radius = int(min(cy, cx, h - cy, w - cx))
    r_int = R.astype(int)
    radii = np.arange(0, max_radius)
    profile = np.zeros(len(radii))
    for ri in radii:
        mask = r_int == ri
        if mask.any():
            profile[ri] = data_2d[mask].mean()
    return radii, profile


def gauss2d(xy, amp, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xy
    a = np.cos(theta)**2 / (2*sigma_x**2) + np.sin(theta)**2 / (2*sigma_y**2)
    b = -np.sin(2*theta) / (4*sigma_x**2) + np.sin(2*theta) / (4*sigma_y**2)
    c_  = np.sin(theta)**2 / (2*sigma_x**2) + np.cos(theta)**2 / (2*sigma_y**2)
    return (offset + amp * np.exp(
        -(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c_*(y-y0)**2))).ravel()


def fit_gaussian(psf):
    h, w = psf.shape
    y, x = np.mgrid[:h, :w]
    cy, cx = subpixel_centre(psf)
    p0 = [psf.max(), cx, cy, 2.0, 2.0, 0.0, 0.0]
    bounds_lo = [0, 0, 0, 0.3, 0.3, -np.pi, -np.inf]
    bounds_hi = [psf.max()*2, w, h, w/2, h/2, np.pi, psf.max()*0.5]
    try:
        popt, _ = curve_fit(gauss2d, (x, y), psf.ravel(), p0=p0,
                            bounds=(bounds_lo, bounds_hi), maxfev=20000)
        return popt   # amp, x0, y0, sx, sy, theta, offset
    except RuntimeError:
        return None


def compute_mtf_from_psf(psf, pixel_pitch_um=None):
    """Radial MTF via 2-D FFT of the PSF. Returns (freq, mtf, freq_label, nyquist)."""
    pad = max(256, psf.shape[0], psf.shape[1])
    padded = np.zeros((pad, pad))
    r0 = (pad - psf.shape[0]) // 2
    c0 = (pad - psf.shape[1]) // 2
    padded[r0:r0 + psf.shape[0], c0:c0 + psf.shape[1]] = psf
    s = padded.sum()
    if s > 0:
        padded /= s
    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    mtf_2d = np.abs(otf)
    mx = mtf_2d.max()
    if mx > 0:
        mtf_2d /= mx
    center = (pad / 2.0, pad / 2.0)
    radii, mtf_profile = radial_average(mtf_2d, center, pad // 2)
    freq_cpp = radii.astype(float) / pad
    if pixel_pitch_um is not None:
        freq = freq_cpp / (pixel_pitch_um * 1e-3)
        freq_label = 'cy/mm'
        nyquist = 1.0 / (2.0 * pixel_pitch_um * 1e-3)
    else:
        freq = freq_cpp
        freq_label = 'cy/px'
        nyquist = 0.5
    return freq, mtf_profile, freq_label, nyquist


def mtf_at_fraction(freq, mtf, frac=0.5):
    above = mtf >= frac
    if not above.any() or above.all():
        return np.nan
    idx = np.where(np.diff(above.astype(int)) == -1)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    f0, f1, m0, m1 = freq[i], freq[i+1], mtf[i], mtf[i+1]
    return f0 if abs(m1-m0) < 1e-12 else f0 + (frac-m0)*(f1-f0)/(m1-m0)


def psf_metrics(psf, pixel_pitch_um=None):
    """Return dict of metrics for one PSF image."""
    popt = fit_gaussian(psf)
    freq, mtf, freq_label, nyquist = compute_mtf_from_psf(psf, pixel_pitch_um)
    mtf50 = mtf_at_fraction(freq, mtf, 0.5)
    mtf10 = mtf_at_fraction(freq, mtf, 0.1)
    result = dict(freq=freq, mtf=mtf, freq_label=freq_label, nyquist=nyquist,
                  mtf50=mtf50, mtf10=mtf10)
    if popt is not None:
        result.update(sigma_x=popt[3], sigma_y=popt[4],
                      fwhm_x=2.355*popt[3], fwhm_y=2.355*popt[4])
    else:
        result.update(sigma_x=np.nan, sigma_y=np.nan,
                      fwhm_x=np.nan, fwhm_y=np.nan)
    return result


# ---------------------------------------------------------------------------
# SR functions  (adapted from sweep_sr.py)
# ---------------------------------------------------------------------------

def shift_and_add(lr_list, shifts_yx, factor=2, order=3):
    """Bicubic upsample + sub-pixel shift + average."""
    h, w = lr_list[0].shape
    acc = np.zeros((h * factor, w * factor))
    for lr, (dy, dx) in zip(lr_list, shifts_yx):
        up = ndi_zoom(lr, factor, order=order)
        acc += ndi_shift(up, (dy * factor, dx * factor), order=3, mode='nearest')
    return acc / len(lr_list)


def forward_model(hr, kernel, shift_yx, factor):
    blurred = fftconvolve(hr, kernel, mode='same')
    shifted = ndi_shift(blurred,
                        (shift_yx[0]*factor, shift_yx[1]*factor),
                        order=3, mode='nearest')
    return shifted[::factor, ::factor]


def back_project(err_lr, kernel, shift_yx, factor, hr_shape):
    h_hr, w_hr = hr_shape
    up = np.zeros((err_lr.shape[0]*factor, err_lr.shape[1]*factor))
    up[::factor, ::factor] = err_lr
    if up.shape[0] < h_hr or up.shape[1] < w_hr:
        up = np.pad(up, ((0, max(0, h_hr-up.shape[0])),
                         (0, max(0, w_hr-up.shape[1]))))
    up = up[:h_hr, :w_hr]
    shifted = ndi_shift(up,
                        (-shift_yx[0]*factor, -shift_yx[1]*factor),
                        order=3, mode='nearest')
    return fftconvolve(shifted, kernel[::-1, ::-1], mode='same')


def ibp(lr_list, shifts_yx, kernel, hr_init, factor=2, n_iter=20, step=0.5):
    hr = hr_init.copy()
    n = len(lr_list)
    errors = []
    for _ in range(n_iter):
        correction = np.zeros_like(hr)
        total_err = 0.0
        for lr, s in zip(lr_list, shifts_yx):
            sim = forward_model(hr, kernel, s, factor)
            mh = min(sim.shape[0], lr.shape[0])
            mw = min(sim.shape[1], lr.shape[1])
            err = lr[:mh, :mw] - sim[:mh, :mw]
            total_err += np.mean(err**2)
            correction += back_project(err, kernel, s, factor, hr.shape)
        hr += step * correction / n
        hr = np.clip(hr, 0, None)
        errors.append(total_err / n)
    return hr, errors


# ---------------------------------------------------------------------------
# Dataset parsing
# ---------------------------------------------------------------------------

def parse_filename(path):
    """Parse image path for both dataset layouts.

    Flat format  (20260325_184308):
        sweepx_tilt0.02000_rep00_pos4.png

    Subfolder format  (20260326_123627):
        sweepx_tilt0.02000deg/pos4_(0,0).png   (repeat always 0)
    """
    name    = os.path.basename(path)
    dirname = os.path.basename(os.path.dirname(path))

    # Flat format
    m = re.match(
        r'sweep([xy])_tilt([\d.]+)_rep(\d+)_pos(\d+)\.(png|tif|tiff|bmp)$',
        name, re.IGNORECASE)
    if m:
        return dict(sweep_axis=m.group(1), tilt_angle=float(m.group(2)),
                    repeat=int(m.group(3)), position=int(m.group(4)), path=path)

    # Subfolder format
    m1 = re.match(r'sweep([xy])_tilt([\d.]+)deg$', dirname, re.IGNORECASE)
    m2 = re.match(r'pos(\d+)[_(].*\.(png|tif|tiff|bmp)$', name, re.IGNORECASE)
    if m1 and m2:
        return dict(sweep_axis=m1.group(1), tilt_angle=float(m1.group(2)),
                    repeat=0, position=int(m2.group(1)), path=path)

    return None


def load_image_index(folder):
    """Return dict  (sweep_axis, tilt_angle, repeat, position) → path.
    Supports both flat (20260325) and subfoldered (20260326) layouts.
    """
    exts = ('*.png', '*.tif', '*.tiff', '*.bmp')
    index = {}
    for ext in exts:
        for p in glob.glob(os.path.join(folder, ext)):           # flat
            rec = parse_filename(p)
            if rec:
                key = (rec['sweep_axis'], round(rec['tilt_angle'], 5),
                       rec['repeat'], rec['position'])
                index[key] = rec['path']
        for p in glob.glob(os.path.join(folder, '*', ext)):      # subfoldered
            rec = parse_filename(p)
            if rec:
                key = (rec['sweep_axis'], round(rec['tilt_angle'], 5),
                       rec['repeat'], rec['position'])
                index[key] = rec['path']
    return index


def load_shifts_csv(folder):
    """Load shifts.csv.
    Returns dict  (sweep_axis, tilt_angle, position) → (dx_mean, dy_mean, dx_std, dy_std).
    """
    path = os.path.join(folder, 'shifts.csv')
    result = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row['sweep_axis'], round(float(row['tilt_angle_deg']), 5),
                   int(row['position']))
            result[key] = (float(row['dx_mean_px']), float(row['dy_mean_px']),
                           float(row['dx_std_px']), float(row['dy_std_px']))
    return result


def load_centers_csv(folder):
    """Load centers.csv.
    Returns dict  (sweep_axis, tilt_angle, repeat, position) → (cx, cy).
    """
    path = os.path.join(folder, 'centers.csv')
    result = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row['sweep_axis'], round(float(row['tilt_angle']), 5),
                   int(row['repeat']), int(row['position']))
            result[key] = (float(row['cx']), float(row['cy']))
    return result


def find_valid_tilts(shifts_data, shift_min_px=0.35, shift_max_px=0.65):
    """Find tilt angles where ALL four SR corner images have ≈0.5 px shift.

    Checks:
      - x-sweep pos2 |dx| ∈ [shift_min, shift_max]
      - y-sweep pos0  dy  ∈ [shift_min, shift_max]
      - y-sweep pos6 |dy| ∈ [shift_min, shift_max]

    Returns sorted list of valid tilt angles.
    """
    # Collect all tilt angles present in the dataset
    all_tilts = sorted(set(k[1] for k in shifts_data))
    valid = []
    for tilt in all_tilts:
        x_pos2 = shifts_data.get(('x', tilt, 2))
        y_pos0 = shifts_data.get(('y', tilt, 0))
        y_pos6 = shifts_data.get(('y', tilt, 6))
        if None in (x_pos2, y_pos0, y_pos6):
            continue
        dx = abs(x_pos2[0])   # x-sweep pos2 dx
        dy_up = y_pos0[1]     # y-sweep pos0 dy (should be positive)
        dy_dn = abs(y_pos6[1])  # y-sweep pos6 |dy|
        if (shift_min_px <= dx <= shift_max_px and
                shift_min_px <= dy_up <= shift_max_px and
                shift_min_px <= dy_dn <= shift_max_px):
            valid.append(tilt)
    return valid


# ---------------------------------------------------------------------------
# PSF kernel builder (for IBP)
# ---------------------------------------------------------------------------

def build_psf_kernel(folder, image_index, psf_halfwidth=4, upsample=2,
                     crop_radius=60, n_repeats=5):
    """Build a PSF kernel from the pos4 reference images (averaged over all
    tilts and available repeats).  Returns the HR (upsampled) kernel array.
    """
    psfs = []
    for key, path in image_index.items():
        axis, tilt, rep, pos = key
        if axis == 'x' and pos == 4:
            img = load_gray(path)
            peak = find_peak(img)
            psf = extract_psf(img, peak, crop_radius)
            target_c = (crop_radius, crop_radius)
            psf = align_psf(psf, target_c)
            psfs.append(psf)

    if not psfs:
        print('WARNING: no pos4 images found; using Gaussian kernel.', file=sys.stderr)
        hw = psf_halfwidth
        y, x = np.mgrid[-hw:hw+1, -hw:hw+1].astype(float)
        kernel = np.exp(-(x**2 + y**2) / (2 * 2.0**2))
        kernel /= kernel.sum()
        return kernel

    psf_avg = np.array(psfs).mean(axis=0)
    cy, cx = psf_avg.shape[0] // 2, psf_avg.shape[1] // 2
    kernel_lr = psf_avg[cy - psf_halfwidth:cy + psf_halfwidth + 1,
                        cx - psf_halfwidth:cx + psf_halfwidth + 1].copy()
    kernel_lr = np.clip(kernel_lr, 0, None)
    if kernel_lr.sum() > 0:
        kernel_lr /= kernel_lr.sum()

    # Upsample kernel to HR pixel scale
    kernel_hr = ndi_zoom(kernel_lr, upsample, order=3)
    kernel_hr = np.clip(kernel_hr, 0, None)
    if kernel_hr.sum() > 0:
        kernel_hr /= kernel_hr.sum()
    return kernel_hr


# ---------------------------------------------------------------------------
# Per-run SR processing
# ---------------------------------------------------------------------------

def run_sr(center_img, shifted_imgs, shifts_nom, shifts_act,
           psf_kernel, upsample=2, crop_radius=60,
           ibp_iterations=20, ibp_step=0.5):
    """
    Apply all SR methods to the (center + 4 shifted) LR images.

    All images are cropped to a region of radius `crop_radius` around the
    PSF peak of the center image before SR.

    Returns a dict  method_name → PSF crop (in HR space for SR methods,
    in LR space for Native-LR).
    """
    # Find crop centre from the reference image
    peak = find_peak(center_img)
    r, c = peak
    h, w = center_img.shape
    r0 = max(r - crop_radius, 0)
    r1 = min(r + crop_radius + 1, h)
    c0 = max(c - crop_radius, 0)
    c1 = min(c + crop_radius + 1, w)

    center_crop = center_img[r0:r1, c0:c1]
    shifted_crops = [img[r0:r1, c0:c1] for img in shifted_imgs]

    f = upsample

    # Native-LR: bicubic upscale of center
    native_hr = ndi_zoom(center_crop, f, order=3)

    # SAA nominal
    saa_nom_hr = shift_and_add(shifted_crops, shifts_nom, f, order=3)

    # SAA actual
    saa_act_hr = shift_and_add(shifted_crops, shifts_act, f, order=3)

    # IBP nominal (initialised from SAA nominal)
    ibp_nom_hr, _ = ibp(shifted_crops, shifts_nom, psf_kernel,
                        saa_nom_hr.copy(), factor=f,
                        n_iter=ibp_iterations, step=ibp_step)

    # IBP actual (initialised from SAA actual)
    ibp_act_hr, _ = ibp(shifted_crops, shifts_act, psf_kernel,
                        saa_act_hr.copy(), factor=f,
                        n_iter=ibp_iterations, step=ibp_step)

    return {
        'Native-LR': center_crop,
        'SAA-nom':   saa_nom_hr,
        'SAA-act':   saa_act_hr,
        'IBP-nom':   ibp_nom_hr,
        'IBP-act':   ibp_act_hr,
    }


def extract_eval_psf(hr_img, crop_radius_hr, is_lr=False, bg_percentile=50.0):
    """Find the PSF in an SR (or LR) output and return it background-subtracted."""
    peak = find_peak(hr_img)
    psf = extract_psf(hr_img, peak, crop_radius_hr, bg_percentile)
    target_c = (crop_radius_hr, crop_radius_hr)
    return align_psf(psf, target_c)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_per_tilt(tilt, results_by_method, pixel_pitch_hr_um, pixel_pitch_lr_um,
                  shifts_nom, shifts_act, upsample, output_path=None):
    """3-row figure for one tilt angle: PSF images, radial PSF, MTF curves."""
    n_methods = len(METHODS)
    zoom_px = 12    # half-width for PSF display in HR pixels

    fig = plt.figure(figsize=(4 * n_methods, 10), constrained_layout=True)
    fig.suptitle('SR evaluation – tilt ≈ %.4f rad  (nom ±0.5 px, act from centers.csv)'
                 % tilt, fontsize=12, fontweight='bold')
    gs = GridSpec(3, n_methods, figure=fig)

    freq_data = {}
    for col, meth in enumerate(METHODS):
        if meth not in results_by_method:
            continue
        psf = results_by_method[meth]['psf']
        metrics = results_by_method[meth]['metrics']
        pitch = pixel_pitch_lr_um if meth == 'Native-LR' else pixel_pitch_hr_um

        # --- Row 0: PSF image ---
        ax = fig.add_subplot(gs[0, col])
        com = subpixel_centre(psf)
        cy_i, cx_i = int(round(com[0])), int(round(com[1]))
        h_p, w_p = psf.shape
        rs = max(0, cy_i - zoom_px); re = min(h_p, cy_i + zoom_px + 1)
        cs = max(0, cx_i - zoom_px); ce = min(w_p, cx_i + zoom_px + 1)
        ax.imshow(psf[rs:re, cs:ce], cmap='inferno', vmin=0, vmax=psf.max(),
                  origin='lower', interpolation='nearest')
        title = meth
        if not np.isnan(metrics['fwhm_x']):
            title += '\nFWHM: %.2f×%.2f px' % (metrics['fwhm_x'] / upsample if meth != 'Native-LR' else metrics['fwhm_x'],
                                                 metrics['fwhm_y'] / upsample if meth != 'Native-LR' else metrics['fwhm_y'])
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        freq_data[meth] = (metrics['freq'], metrics['mtf'],
                           metrics['nyquist'], metrics['mtf50'])

    # --- Row 1: Radial MTF curves ---
    ax_mtf = fig.add_subplot(gs[1, :])
    nyquist_lr = 1.0 / (2.0 * pixel_pitch_lr_um * 1e-3)
    nyquist_hr = 1.0 / (2.0 * pixel_pitch_hr_um * 1e-3)
    for meth in METHODS:
        if meth not in freq_data:
            continue
        freq, mtf, nyq, mtf50 = freq_data[meth]
        valid = freq <= nyquist_hr
        lw = 2.0 if meth == 'Native-LR' else 1.2
        ls = '-' if meth in ('Native-LR', 'SAA-nom', 'IBP-nom') else '--'
        ax_mtf.plot(freq[valid], mtf[valid], color=COLORS.get(meth, 'gray'),
                    lw=lw, ls=ls,
                    label='%s  (MTF50=%.1f cy/mm)' % (meth, mtf50 if not np.isnan(mtf50) else 0))
    ax_mtf.axvline(nyquist_lr, color='gray', ls=':', alpha=0.5, label='LR Nyquist')
    ax_mtf.axvline(nyquist_hr, color='gray', ls='--', alpha=0.5, label='HR Nyquist')
    ax_mtf.axhline(0.5, color='orange', ls='--', alpha=0.4)
    ax_mtf.set_xlabel('Spatial frequency (cy/mm)')
    ax_mtf.set_ylabel('MTF')
    ax_mtf.set_title('PSF-based radial MTF')
    ax_mtf.set_xlim(0, nyquist_hr * 1.05)
    ax_mtf.set_ylim(0, 1.05)
    ax_mtf.legend(fontsize=7, ncol=2)
    ax_mtf.grid(True, alpha=0.3)

    # --- Row 2: shift summary text ---
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis('off')
    lines = ['Nominal shifts (dy, dx) in LR px:']
    for (axis, pos, snom), sact in zip(SR_CORNERS, shifts_act):
        lines.append('  sweep-%s pos%d  nom=(%.2f, %.2f)  act=(%.3f, %.3f)'
                     % (axis, pos, snom[0], snom[1], sact[0], sact[1]))
    lines.append('')
    lines.append('MTF50 summary (cy/mm):')
    for meth in METHODS:
        if meth in results_by_method:
            m = results_by_method[meth]['metrics']
            lines.append('  %-12s  MTF50=%.1f   FWHM_x=%.2f px   FWHM_y=%.2f px'
                         % (meth,
                            m['mtf50'] if not np.isnan(m['mtf50']) else 0,
                            m['fwhm_x'] if not np.isnan(m['fwhm_x']) else 0,
                            m['fwhm_y'] if not np.isnan(m['fwhm_y']) else 0))
    ax_txt.text(0.02, 0.95, '\n'.join(lines), transform=ax_txt.transAxes,
                fontsize=7.5, va='top', fontfamily='monospace')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print('  Saved %s' % output_path)
    plt.close(fig)


def plot_summary(all_results, pixel_pitch_hr_um, pixel_pitch_lr_um, output_path=None):
    """MTF50, FWHM vs tilt angle for each method, with repeat spread."""
    tilts = sorted(all_results.keys())
    if not tilts:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle('SR evaluation summary across tilt angles', fontsize=13, fontweight='bold')

    for meth in METHODS:
        mtf50_per_tilt = []
        fwhm_per_tilt = []
        for tilt in tilts:
            reps = all_results[tilt]
            mtf50s = [r[meth]['metrics']['mtf50']
                      for r in reps if meth in r and not np.isnan(r[meth]['metrics']['mtf50'])]
            fwhms = [(r[meth]['metrics']['fwhm_x'] + r[meth]['metrics']['fwhm_y']) / 2.0
                     for r in reps if meth in r]
            fwhms = [f for f in fwhms if not np.isnan(f)]
            mtf50_per_tilt.append(mtf50s)
            fwhm_per_tilt.append(fwhms)

        means_mtf = [np.mean(v) if v else np.nan for v in mtf50_per_tilt]
        stds_mtf  = [np.std(v)  if len(v) > 1 else 0.0 for v in mtf50_per_tilt]
        means_fwhm = [np.mean(v) if v else np.nan for v in fwhm_per_tilt]
        stds_fwhm  = [np.std(v)  if len(v) > 1 else 0.0 for v in fwhm_per_tilt]

        col = COLORS.get(meth, 'gray')
        ls  = '-' if meth in ('Native-LR', 'SAA-nom', 'IBP-nom') else '--'
        lw  = 2.0 if meth == 'Native-LR' else 1.2

        axes[0].errorbar(tilts, means_mtf, yerr=stds_mtf,
                         color=col, ls=ls, lw=lw, marker='o', ms=5, label=meth)
        axes[1].errorbar(tilts, means_fwhm, yerr=stds_fwhm,
                         color=col, ls=ls, lw=lw, marker='o', ms=5, label=meth)

    # Improvement ratio (best SR / Native-LR MTF50)
    for meth in METHODS:
        if meth == 'Native-LR':
            continue
        ratios = []
        for tilt in tilts:
            reps = all_results[tilt]
            sr_m = [r[meth]['metrics']['mtf50'] for r in reps if meth in r]
            lr_m = [r['Native-LR']['metrics']['mtf50'] for r in reps if 'Native-LR' in r]
            if sr_m and lr_m and not np.isnan(np.mean(sr_m)) and not np.isnan(np.mean(lr_m)):
                ratios.append(np.mean(sr_m) / np.mean(lr_m))
            else:
                ratios.append(np.nan)
        axes[2].plot(tilts, ratios, color=COLORS.get(meth, 'gray'),
                     ls='--' if 'act' in meth.lower() else '-',
                     lw=1.2, marker='o', ms=5, label=meth)

    nyquist_lr = 1.0 / (2.0 * pixel_pitch_lr_um * 1e-3)
    axes[0].axhline(nyquist_lr, color='gray', ls=':', alpha=0.5, label='LR Nyquist')
    axes[0].set_xlabel('Tilt angle (rad)'); axes[0].set_ylabel('MTF50 (cy/mm)')
    axes[0].set_title('MTF50 vs tilt angle')
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Tilt angle (rad)'); axes[1].set_ylabel('Mean FWHM (px, in native space)')
    axes[1].set_title('Mean FWHM vs tilt angle\n(HR FWHM ÷ upsample for SR methods)')
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    axes[2].axhline(1.0, color='gray', ls=':', alpha=0.5)
    axes[2].set_xlabel('Tilt angle (rad)'); axes[2].set_ylabel('MTF50 ratio (method / Native-LR)')
    axes[2].set_title('MTF50 improvement over Native-LR')
    axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print('Saved summary figure: %s' % output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='PSF-based SR evaluation')
    parser.add_argument('folder', help='Path to 20260325_184308 dataset folder')
    parser.add_argument('--pixel-pitch-um', type=float, default=PIXEL_PITCH_UM)
    parser.add_argument('--upsample', type=int, default=2)
    parser.add_argument('--shift-min-px', type=float, default=0.35,
                        help='Min corner shift to include (default: 0.35)')
    parser.add_argument('--shift-max-px', type=float, default=0.65,
                        help='Max corner shift to include (default: 0.65)')
    parser.add_argument('--crop-radius', type=int, default=60,
                        help='LR pixel radius around PSF for SR (default: 60)')
    parser.add_argument('--psf-halfwidth', type=int, default=4,
                        help='PSF kernel half-width (default: 4)')
    parser.add_argument('--ibp-iterations', type=int, default=20)
    parser.add_argument('--ibp-step', type=float, default=0.5)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    folder = args.folder
    f      = args.upsample
    pitch_lr = args.pixel_pitch_um          # µm / LR pixel
    pitch_hr = pitch_lr / f                 # µm / HR pixel

    out_dir = args.output_dir or os.path.join(folder, 'sr_results')
    os.makedirs(out_dir, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print('Indexing images ...')
    img_index = load_image_index(folder)
    print('  %d images found' % len(img_index))

    print('Loading shifts.csv / centers.csv ...')
    shifts_data  = load_shifts_csv(folder)
    centers_data = load_centers_csv(folder)

    # ── Valid tilt angles ────────────────────────────────────────────────────
    valid_tilts = find_valid_tilts(shifts_data, args.shift_min_px, args.shift_max_px)
    if not valid_tilts:
        sys.exit('No tilt angles found with corner shifts in [%.2f, %.2f] px'
                 % (args.shift_min_px, args.shift_max_px))
    print('\nTilt angles with corner shifts in [%.2f, %.2f] px:' %
          (args.shift_min_px, args.shift_max_px))
    for t in valid_tilts:
        x2  = shifts_data.get(('x', t, 2), (np.nan,)*4)
        y0  = shifts_data.get(('y', t, 0), (np.nan,)*4)
        y6  = shifts_data.get(('y', t, 6), (np.nan,)*4)
        print('  tilt=%.5f  x-pos2 dx=%.3f  y-pos0 dy=%.3f  y-pos6 dy=%.3f'
              % (t, x2[0], y0[1], y6[1]))

    # ── Build PSF kernel (for IBP) ───────────────────────────────────────────
    print('\nBuilding PSF kernel from pos4 reference images ...')
    psf_kernel = build_psf_kernel(folder, img_index,
                                  psf_halfwidth=args.psf_halfwidth,
                                  upsample=f, crop_radius=args.crop_radius)
    print('  HR kernel shape: %s' % str(psf_kernel.shape))

    # ── Determine available repeats ──────────────────────────────────────────
    available_repeats = sorted(set(k[2] for k in img_index))
    print('\nAvailable repeats: %s' % available_repeats)

    # ── Process each tilt angle ──────────────────────────────────────────────
    all_results   = {}   # tilt → list-of-repeat-results
    crop_r_hr     = args.crop_radius * f  # crop radius in HR pixels

    for tilt in valid_tilts:
        print('\n=== tilt = %.5f rad ===' % tilt)
        tilt_results = []

        for rep in available_repeats:
            # -- Check all required images exist --
            center_key = (SR_CENTER[0], tilt, rep, SR_CENTER[1])
            corner_keys = [(ax, tilt, rep, pos) for ax, pos, _ in SR_CORNERS]
            if center_key not in img_index:
                continue
            if any(k not in img_index for k in corner_keys):
                continue

            # -- Load images --
            center_img   = load_gray(img_index[center_key])
            shifted_imgs = [load_gray(img_index[k]) for k in corner_keys]

            # -- Actual shifts from centers.csv --
            # shifts_yx = (cy_k - cy_ref, cx_k - cx_ref)  in LR pixels
            ref_cx, ref_cy = centers_data.get(center_key, (np.nan, np.nan))
            shifts_act = []
            for k in corner_keys:
                cx_k, cy_k = centers_data.get(k, (ref_cx, ref_cy))
                shifts_act.append((cy_k - ref_cy, cx_k - ref_cx))

            # -- Nominal shifts --
            shifts_nom = [s for _, _, s in SR_CORNERS]

            print('  rep=%02d  act shifts (dy,dx): %s'
                  % (rep, ['(%.3f,%.3f)' % s for s in shifts_act]))

            # -- Run SR --
            hr_crops = run_sr(
                center_img, shifted_imgs, shifts_nom, shifts_act,
                psf_kernel, upsample=f, crop_radius=args.crop_radius,
                ibp_iterations=args.ibp_iterations, ibp_step=args.ibp_step,
            )

            # -- Evaluate each output --
            rep_result = {}
            for meth, img in hr_crops.items():
                is_lr = (meth == 'Native-LR')
                cr    = args.crop_radius if is_lr else crop_r_hr
                pitch = pitch_lr if is_lr else pitch_hr
                psf_eval = extract_eval_psf(img, cr, is_lr=is_lr)
                metrics  = psf_metrics(psf_eval, pixel_pitch_um=pitch)

                # For display: convert HR FWHM to native-pixel units
                if not is_lr:
                    metrics['fwhm_x'] /= f
                    metrics['fwhm_y'] /= f
                    metrics['sigma_x'] /= f
                    metrics['sigma_y'] /= f

                rep_result[meth] = {'psf': psf_eval, 'metrics': metrics}
                print('    %-12s  MTF50=%6.1f cy/mm  FWHM=(%.2f,%.2f) px'
                      % (meth,
                         metrics['mtf50'] if not np.isnan(metrics['mtf50']) else 0,
                         metrics['fwhm_x'] if not np.isnan(metrics['fwhm_x']) else 0,
                         metrics['fwhm_y'] if not np.isnan(metrics['fwhm_y']) else 0))

            tilt_results.append(rep_result)

        all_results[tilt] = tilt_results

        # -- Per-tilt figure (average-rep result for display) --
        if tilt_results:
            # Compute actual shifts averaged over repeats at this tilt for display
            x2 = shifts_data.get(('x', tilt, 2), (0.0, 0.0, 0.0, 0.0))
            y0 = shifts_data.get(('y', tilt, 0), (0.0, 0.0, 0.0, 0.0))
            y6 = shifts_data.get(('y', tilt, 6), (0.0, 0.0, 0.0, 0.0))
            x0 = shifts_data.get(('x', tilt, 0), (0.0, 0.0, 0.0, 0.0))
            shifts_act_avg = [
                (0.0,       x0[0]),   # x-sweep pos0: (dy≈0, dx)
                (0.0,       x2[0]),   # x-sweep pos2: (dy≈0, dx)
                (y0[1],     0.0),     # y-sweep pos0: (dy, dx≈0)
                (y6[1],     0.0),     # y-sweep pos6: (dy, dx≈0)
            ]
            # Use first-repeat results for the figure
            plot_per_tilt(
                tilt,
                tilt_results[0],        # first repeat
                pixel_pitch_hr_um=pitch_hr,
                pixel_pitch_lr_um=pitch_lr,
                shifts_nom=shifts_nom,
                shifts_act=shifts_act_avg,
                upsample=f,
                output_path=os.path.join(out_dir,
                                         'psf_sr_tilt%.5f.png' % tilt),
            )

    # ── Summary figure ───────────────────────────────────────────────────────
    plot_summary(all_results, pitch_hr, pitch_lr,
                 output_path=os.path.join(out_dir, 'psf_sr_summary.png'))

    # ── Save numerical data ──────────────────────────────────────────────────
    save_dict = {}
    save_dict['tilts']   = np.array(valid_tilts)
    save_dict['methods'] = np.array(METHODS)

    for tilt in valid_tilts:
        tilt_key = 'tilt_%.5f' % tilt
        for meth in METHODS:
            prefix = '%s_%s_' % (tilt_key, meth.replace('-', '_'))
            mtf50s  = [r[meth]['metrics']['mtf50']
                       for r in all_results[tilt] if meth in r]
            mtf10s  = [r[meth]['metrics']['mtf10']
                       for r in all_results[tilt] if meth in r]
            fwhm_xs = [r[meth]['metrics']['fwhm_x']
                       for r in all_results[tilt] if meth in r]
            fwhm_ys = [r[meth]['metrics']['fwhm_y']
                       for r in all_results[tilt] if meth in r]
            save_dict[prefix + 'mtf50']  = np.array(mtf50s)
            save_dict[prefix + 'mtf10']  = np.array(mtf10s)
            save_dict[prefix + 'fwhm_x'] = np.array(fwhm_xs)
            save_dict[prefix + 'fwhm_y'] = np.array(fwhm_ys)

            # Also save the MTF curves from the first repeat
            reps = all_results[tilt]
            if reps and meth in reps[0]:
                m = reps[0][meth]['metrics']
                save_dict[prefix + 'freq'] = m['freq']
                save_dict[prefix + 'mtf']  = m['mtf']

    npz_path = os.path.join(out_dir, 'psf_sr_data.npz')
    np.savez(npz_path, **save_dict)
    print('\nSaved numerical data to %s' % npz_path)

    # ── Console summary table ────────────────────────────────────────────────
    print('\n' + '─' * 75)
    print('%-10s ' % 'Tilt', end='')
    for m in METHODS:
        print('%13s' % m, end='')
    print()
    print('─' * 75)
    for tilt in valid_tilts:
        reps = all_results[tilt]
        if not reps:
            continue
        print('%-10.5f ' % tilt, end='')
        for meth in METHODS:
            vals = [r[meth]['metrics']['mtf50']
                    for r in reps if meth in r and not np.isnan(r[meth]['metrics']['mtf50'])]
            if vals:
                print('%9.1f±%-3.1f' % (np.mean(vals), np.std(vals)), end='')
            else:
                print('%13s' % 'N/A', end='')
        print()
    print('─' * 75)
    print('MTF50 in cy/mm (mean ± std over repeats) — higher is better')
    print('\nOutputs saved to: %s' % out_dir)


if __name__ == '__main__':
    main()
