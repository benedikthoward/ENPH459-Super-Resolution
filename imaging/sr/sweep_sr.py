#!/usr/bin/env python3
"""
sweep_sr.py

Sweep all tilt/settle combinations in 20260318_203508, run all SR methods,
evaluate quality metrics, and produce per-combination and summary results.

Speed strategy: IBP and TV-SR run only on a cropped ROI containing the star
and edge targets (~5.8× smaller than the full image), dramatically reducing
runtime compared to the full 1536×2048 frame.
"""

import os
import json
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import shift as ndi_shift, gaussian_filter, zoom as ndi_zoom
from scipy.signal import fftconvolve, argrelextrema
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 100})

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(SCRIPT_DIR, '20260318_203508')
RESULTS_DIR   = os.path.join(DATA_DIR, 'results')
PSF_DATA_PATH = os.path.join(SCRIPT_DIR, 'cal', 'psf_mtf_results_data_pos0.npz')

# ── Sensor / optics ────────────────────────────────────────────────────────────

PIXEL_PITCH_UM  = 3.45
UPSAMPLE_FACTOR = 2
f               = UPSAMPLE_FACTOR
PITCH_LR_MM     = PIXEL_PITCH_UM * 1e-3
PITCH_HR_MM     = PITCH_LR_MM / f
NYQUIST_LR      = 0.5 / PITCH_LR_MM    # cy/mm
NYQUIST_HR      = 0.5 / PITCH_HR_MM

# Nominal shifts: always ±0.5 px diagonal, regardless of tilt angle being swept
NOMINAL_SHIFTS = [
    (-0.5, -0.5),   # shift_0
    (+0.5, -0.5),   # shift_1
    (-0.5, +0.5),   # shift_2
    (+0.5, +0.5),   # shift_3
]

# ── ROI definitions (LR pixel coordinates in the full image) ──────────────────

EDGE_ROI_LR     = (550, 720, 580, 590)  # (r0, r1, c0, c1) — slanted edge
EDGE_ORIENT     = 'vertical'
STAR_CENTER_LR  = (804, 978)            # (row, col) — ISO 12233 star
STAR_ROI_R      = 360

RING_ANGLE_MIN  = 230   # degrees
RING_ANGLE_MAX  = 30    # degrees (wraps through 0°)
RING_R_MIN_LR   = 60
RING_R_MAX_LR   = 360

# Combined LR crop: tight bounding box around star + edge targets.
# Star spans rows [804-360, 804+360] = [444, 1164], cols [978-360, 978+360] = [618, 1338]
# Edge spans rows [550, 720], cols [580, 590]
# Union (with a small margin on the left for the edge cols):
CROP_R0, CROP_R1 = 444, 1164
CROP_C0, CROP_C1 = 580, 1338

# Derived ROIs in crop-local LR coordinates
EDGE_ROI_CROP = (
    EDGE_ROI_LR[0] - CROP_R0,
    EDGE_ROI_LR[1] - CROP_R0,
    EDGE_ROI_LR[2] - CROP_C0,
    EDGE_ROI_LR[3] - CROP_C0,
)
STAR_CENTER_CROP = (
    STAR_CENTER_LR[0] - CROP_R0,
    STAR_CENTER_LR[1] - CROP_C0,
)

# ── Iterative method parameters ───────────────────────────────────────────────

PSF_HALFWIDTH  = 3
IBP_ITERATIONS = 20
IBP_STEP_SIZE  = 0.5
TV_ITERATIONS  = 20
TV_STEP_SIZE   = 0.15
TV_LAMBDA      = 0.005

COLORS = {'Native': 'C0', 'SAA': 'C2', 'IBP': 'C3', 'TV-SR': 'C4'}
METHODS = ['Native', 'SAA', 'IBP', 'TV-SR']

# ── Core image functions ───────────────────────────────────────────────────────

def load_gray(path):
    img = np.array(Image.open(path), dtype=np.float64)
    return img.mean(axis=2) if img.ndim == 3 else img


def blur(img, kernel):
    """Convolve with PSF using FFT (faster than direct conv for large images)."""
    return fftconvolve(img, kernel, mode='same')


def forward_model(hr, kernel, shift_yx, factor):
    """HR → blur → sub-pixel shift → decimate."""
    blurred = blur(hr, kernel)
    shifted = ndi_shift(blurred,
                        (shift_yx[0] * factor, shift_yx[1] * factor),
                        order=3, mode='nearest')
    return shifted[::factor, ::factor]


def back_project(error_lr, kernel, shift_yx, factor, hr_shape):
    """Adjoint of forward_model: LR error → HR correction."""
    h_hr, w_hr = hr_shape
    up = np.zeros((error_lr.shape[0] * factor, error_lr.shape[1] * factor))
    up[::factor, ::factor] = error_lr
    if up.shape[0] < h_hr or up.shape[1] < w_hr:
        up = np.pad(up, ((0, max(0, h_hr - up.shape[0])),
                         (0, max(0, w_hr - up.shape[1]))))
    up = up[:h_hr, :w_hr]
    shifted = ndi_shift(up,
                        (-shift_yx[0] * factor, -shift_yx[1] * factor),
                        order=3, mode='nearest')
    return blur(shifted, kernel[::-1, ::-1])


def shift_and_add(lr_list, shifts_yx, factor=2, order=3):
    h_lr, w_lr = lr_list[0].shape
    acc = np.zeros((h_lr * factor, w_lr * factor))
    for lr, (dy, dx) in zip(lr_list, shifts_yx):
        up = ndi_zoom(lr, factor, order=order)
        acc += ndi_shift(up, (dy * factor, dx * factor), order=3, mode='nearest')
    return acc / len(lr_list)


def ibp(lr_list, shifts_yx, kernel, hr_init, factor=2, n_iter=10, step=0.5):
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
            total_err += np.mean(err ** 2)
            correction += back_project(err, kernel, s, factor, hr.shape)
        hr += step * correction / n
        hr = np.clip(hr, 0, 255)
        errors.append(total_err / n)
    return hr, errors


def tv_gradient(img):
    eps = 1e-8
    dy = np.zeros_like(img); dy[:-1, :] = img[1:, :] - img[:-1, :]
    dx = np.zeros_like(img); dx[:, :-1] = img[:, 1:] - img[:, :-1]
    mag = np.sqrt(dy ** 2 + dx ** 2 + eps)
    ny, nx = dy / mag, dx / mag
    div = np.zeros_like(img)
    div[1:, :] -= ny[:-1, :];  div[:-1, :] += ny[:-1, :]
    div[:, 1:] -= nx[:, :-1];  div[:, :-1] += nx[:, :-1]
    return -div


def tv_sr(lr_list, shifts_yx, kernel, hr_init, factor=2, n_iter=50, step=0.1, lam=0.01):
    hr = hr_init.copy()
    n = len(lr_list)
    errors = []
    for _ in range(n_iter):
        data_grad = np.zeros_like(hr)
        total_err = 0.0
        for lr, s in zip(lr_list, shifts_yx):
            sim = forward_model(hr, kernel, s, factor)
            mh = min(sim.shape[0], lr.shape[0])
            mw = min(sim.shape[1], lr.shape[1])
            residual = sim[:mh, :mw] - lr[:mh, :mw]
            total_err += np.mean(residual ** 2)
            data_grad += back_project(residual, kernel, s, factor, hr.shape)
        data_grad /= n
        hr -= step * (data_grad + lam * tv_gradient(hr))
        hr = np.clip(hr, 0, 255)
        errors.append(total_err / n)
    return hr, errors

# ── Metric functions ───────────────────────────────────────────────────────────

def slanted_edge_mtf(img, roi, orientation='vertical', oversample=4):
    r0, r1, c0, c1 = roi
    patch = img[r0:r1, c0:c1].copy()
    if orientation == 'horizontal':
        patch = patch.T
    rows, cols = patch.shape

    edge_pts = []
    for r in range(rows):
        d = np.abs(np.diff(patch[r, :].astype(float)))
        if d.max() < 5:
            continue
        peaks = np.where(d > d.max() * 0.5)[0]
        if len(peaks):
            edge_pts.append((r, np.average(peaks, weights=d[peaks])))
    if len(edge_pts) < 10:
        return None, None

    edge_pts = np.array(edge_pts)
    slope, intercept = np.polyfit(edge_pts[:, 0], edge_pts[:, 1], 1)
    cos_a = np.cos(np.arctan(slope))

    dists, vals = [], []
    for r in range(rows):
        edge_c = slope * r + intercept
        for c in range(cols):
            dists.append((c - edge_c) * cos_a)
            vals.append(patch[r, c])

    dists = np.array(dists)
    vals  = np.array(vals)
    bw    = 1.0 / oversample
    bins  = np.arange(dists.min(), dists.max(), bw)
    esf   = np.zeros(len(bins))
    cnt   = np.zeros(len(bins))
    idx   = np.clip(((dists - dists.min()) / bw).astype(int), 0, len(bins) - 1)
    np.add.at(esf, idx, vals)
    np.add.at(cnt, idx, 1)
    valid = cnt > 0
    esf[valid] /= cnt[valid]
    esf[~valid] = np.interp(bins[~valid], bins[valid], esf[valid])

    lsf   = np.diff(gaussian_filter(esf, sigma=0.5))
    lsf_w = lsf * np.hanning(len(lsf))
    mtf   = np.abs(np.fft.rfft(lsf_w))
    mtf  /= mtf[0]
    freq  = np.fft.rfftfreq(len(lsf_w), d=bw)

    return freq, mtf


def compute_mtf50(freq_cymm, mtf):
    """MTF50 in cy/mm via linear interpolation."""
    if freq_cymm is None or len(freq_cymm) < 2:
        return float('nan')
    for i in range(len(mtf) - 1):
        if mtf[i] >= 0.5 >= mtf[i + 1]:
            t = (0.5 - mtf[i]) / (mtf[i + 1] - mtf[i])
            return float(freq_cymm[i] + t * (freq_cymm[i + 1] - freq_cymm[i]))
    return float('nan')


def get_ring_angles(angle_min, angle_max, step=3):
    if angle_min > angle_max:
        return list(range(angle_min, 360, step)) + list(range(0, angle_max + 1, step))
    return list(range(angle_min, angle_max + 1, step))


def radial_profile(img, center_rc, theta_rad, max_r):
    cy, cx = center_rc
    rs = np.arange(max_r)
    ys = np.clip(np.round(cy + rs * np.sin(theta_rad)).astype(int), 0, img.shape[0] - 1)
    xs = np.clip(np.round(cx + rs * np.cos(theta_rad)).astype(int), 0, img.shape[1] - 1)
    return img[ys, xs].astype(float)


def radial_ring_contrast(img, center_rc, r_min, r_max,
                          angle_min=RING_ANGLE_MIN, angle_max=RING_ANGLE_MAX,
                          smooth_sigma=1.0, peak_order=3):
    degs      = get_ring_angles(angle_min, angle_max)
    max_r     = r_max + 5
    all_pairs = []
    for d in degs:
        theta = np.radians(d)
        prof  = radial_profile(img, center_rc, theta, max_r)
        prof_s = gaussian_filter1d(prof, sigma=smooth_sigma)
        peaks   = argrelextrema(prof_s, np.greater, order=peak_order)[0]
        valleys = argrelextrema(prof_s, np.less,    order=peak_order)[0]
        peaks   = peaks[(peaks >= r_min) & (peaks < r_max)]
        valleys = valleys[(valleys >= r_min) & (valleys < r_max)]
        extrema = sorted(
            [(p, prof_s[p], 'P') for p in peaks] +
            [(v, prof_s[v], 'V') for v in valleys],
            key=lambda x: x[0]
        )
        for i in range(len(extrema) - 1):
            r1, v1, t1 = extrema[i]
            r2, v2, t2 = extrema[i + 1]
            if t1 != t2:
                hi, lo = max(v1, v2), min(v1, v2)
                if (hi + lo) > 10 and (hi - lo) > 5:
                    all_pairs.append(((r1 + r2) / 2, (hi - lo) / (hi + lo), abs(r2 - r1)))

    if not all_pairs:
        return np.array([]), np.array([])
    arr = np.array(all_pairs)
    return arr[:, 0], arr[:, 1]   # radii (px), contrast


# ── PSF loading ────────────────────────────────────────────────────────────────

def load_psf():
    data     = np.load(PSF_DATA_PATH, allow_pickle=True)
    psf_full = data['psf_avg']
    cy, cx   = psf_full.shape[0] // 2, psf_full.shape[1] // 2
    kernel   = psf_full[cy - PSF_HALFWIDTH:cy + PSF_HALFWIDTH + 1,
                        cx - PSF_HALFWIDTH:cx + PSF_HALFWIDTH + 1].copy()
    kernel   = np.clip(kernel, 0, None)
    kernel  /= kernel.sum()
    kernel_hr = ndi_zoom(kernel, f, order=3)
    kernel_hr = np.clip(kernel_hr, 0, None)
    kernel_hr /= kernel_hr.sum()
    return kernel_hr   # only HR kernel needed for iterative methods


# ── Per-combination processing ─────────────────────────────────────────────────

def process_combination(combo_dir, psf_hr, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, 'metrics.json')

    if os.path.exists(metrics_path):
        print('    [skip] already done')
        with open(metrics_path) as fp:
            return json.load(fp)

    # Load and crop all images to the combined ROI
    center_lr = load_gray(os.path.join(combo_dir, 'center.png'))
    shifts_lr = [load_gray(os.path.join(combo_dir, f'shift_{i}.png')) for i in range(4)]

    center_crop = center_lr[CROP_R0:CROP_R1, CROP_C0:CROP_C1]
    shifts_crop = [s[CROP_R0:CROP_R1, CROP_C0:CROP_C1] for s in shifts_lr]

    # Native: bicubic upscale of centre frame
    native_hr = ndi_zoom(center_crop, f, order=3)

    # SAA bicubic
    saa_hr = shift_and_add(shifts_crop, NOMINAL_SHIFTS, f, order=3)

    # IBP (initialised from SAA)
    print('    IBP  ...', end=' ', flush=True)
    t0 = time.time()
    ibp_hr, ibp_errors = ibp(
        shifts_crop, NOMINAL_SHIFTS, psf_hr, saa_hr.copy(),
        factor=f, n_iter=IBP_ITERATIONS, step=IBP_STEP_SIZE,
    )
    print(f'{time.time() - t0:.0f}s')

    # TV-SR (initialised from SAA)
    print('    TV-SR...', end=' ', flush=True)
    t0 = time.time()
    tvsr_hr, tv_errors = tv_sr(
        shifts_crop, NOMINAL_SHIFTS, psf_hr, saa_hr.copy(),
        factor=f, n_iter=TV_ITERATIONS, step=TV_STEP_SIZE, lam=TV_LAMBDA,
    )
    print(f'{time.time() - t0:.0f}s')

    hr_images = {
        'Native': native_hr,
        'SAA':    saa_hr,
        'IBP':    ibp_hr,
        'TV-SR':  tvsr_hr,
    }

    # ── Slanted-edge MTF metrics (all in HR space) ──
    roi_hr = tuple(x * f for x in EDGE_ROI_CROP)
    mtf_metrics = {}
    for name, img in hr_images.items():
        freq, mtf = slanted_edge_mtf(img, roi_hr, EDGE_ORIENT)
        freq_cymm = freq / PITCH_HR_MM if freq is not None else None
        mtf_metrics[name] = {'mtf50_cymm': compute_mtf50(freq_cymm, mtf)}

    # Also compute MTF50 for the native LR crop directly (true LR reference)
    freq_lr, mtf_lr = slanted_edge_mtf(center_crop, EDGE_ROI_CROP, EDGE_ORIENT)
    mtf_metrics['Native_LR'] = {
        'mtf50_cymm': compute_mtf50(
            freq_lr / PITCH_LR_MM if freq_lr is not None else None,
            mtf_lr,
        )
    }

    # ── Ring contrast metrics (HR) ──
    star_hr = (STAR_CENTER_CROP[0] * f, STAR_CENTER_CROP[1] * f)
    r_min_hr = RING_R_MIN_LR * f
    r_max_hr = RING_R_MAX_LR * f
    contrast_metrics = {}
    for name, img in hr_images.items():
        radii, contrasts = radial_ring_contrast(
            img, star_hr, r_min_hr, r_max_hr,
            peak_order=4,
        )
        contrast_metrics[name] = {
            'mean_contrast': float(np.mean(contrasts)) if len(contrasts) else float('nan'),
            'n_pairs':       int(len(contrasts)),
        }

    metrics = {
        'mtf':          mtf_metrics,
        'contrast':     contrast_metrics,
        'ibp_final_mse': float(ibp_errors[-1]) if ibp_errors else float('nan'),
        'tv_final_mse':  float(tv_errors[-1])  if tv_errors  else float('nan'),
    }

    with open(metrics_path, 'w') as fp:
        json.dump(metrics, fp, indent=2)

    # ── Save plots ──
    _save_comparison_plot(hr_images, star_hr, out_dir)
    _save_mtf_plot(hr_images, roi_hr, out_dir)
    _save_convergence_plot(ibp_errors, tv_errors, out_dir)

    return metrics


def _save_comparison_plot(hr_images, star_hr, out_dir):
    crop_r = RING_R_MAX_LR * f
    cy, cx = int(star_hr[0]), int(star_hr[1])
    r0 = max(0, cy - crop_r); r1 = min(list(hr_images.values())[0].shape[0], cy + crop_r)
    c0 = max(0, cx - crop_r); c1 = min(list(hr_images.values())[0].shape[1], cx + crop_r)

    n = len(hr_images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    for ax, (name, img) in zip(axes, hr_images.items()):
        ax.imshow(img[r0:r1, c0:c1], cmap='gray', interpolation='nearest')
        ax.set_title(name, fontsize=10)
        ax.axis('off')
    plt.suptitle('Star pattern — method comparison', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison.png'), bbox_inches='tight')
    plt.close(fig)


def _save_mtf_plot(hr_images, roi_hr, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, img in hr_images.items():
        freq, mtf = slanted_edge_mtf(img, roi_hr, EDGE_ORIENT)
        if freq is not None:
            ax.plot(freq / PITCH_HR_MM, mtf,
                    color=COLORS.get(name, 'gray'), lw=1.5, label=name)
    ax.axvline(NYQUIST_LR, ls='--', color='gray', alpha=0.5, label='LR Nyquist')
    ax.axvline(NYQUIST_HR, ls=':',  color='gray', alpha=0.5, label='HR Nyquist')
    ax.axhline(0.5, ls='--', color='orange', alpha=0.4, label='MTF50')
    ax.set_xlabel('Spatial frequency (cy/mm)')
    ax.set_ylabel('MTF')
    ax.set_xlim(0, NYQUIST_HR)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Slanted-Edge MTF')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mtf_curves.png'), bbox_inches='tight')
    plt.close(fig)


def _save_convergence_plot(ibp_errors, tv_errors, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ibp_errors, color=COLORS['IBP'], lw=1.5)
    axes[0].set_title('IBP convergence'); axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('MSE'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(tv_errors, color=COLORS['TV-SR'], lw=1.5)
    axes[1].set_title('TV-SR convergence'); axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('MSE'); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'convergence.png'), bbox_inches='tight')
    plt.close(fig)


# ── Summary ────────────────────────────────────────────────────────────────────

def generate_summary(all_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Collect unique sorted tilt/settle values
    tilts   = sorted(set(float(k.split('|')[0]) for k in all_results))
    settles = sorted(set(float(k.split('|')[1]) for k in all_results))

    # Build 2D grids: rows=tilt, cols=settle
    mtf50_grids    = {m: np.full((len(tilts), len(settles)), np.nan) for m in METHODS}
    contrast_grids = {m: np.full((len(tilts), len(settles)), np.nan) for m in METHODS}

    for key, metrics in all_results.items():
        tilt_str, settle_str = key.split('|')
        ti = tilts.index(float(tilt_str))
        si = settles.index(float(settle_str))
        for m in METHODS:
            if m in metrics.get('mtf', {}):
                mtf50_grids[m][ti, si] = metrics['mtf'][m].get('mtf50_cymm', np.nan)
            if m in metrics.get('contrast', {}):
                contrast_grids[m][ti, si] = metrics['contrast'][m].get('mean_contrast', np.nan)

    settle_labels = [f'{s:.0f} ms' for s in settles]
    tilt_labels   = [f'{t:.5f}' for t in tilts]

    # ── Heatmap figure: MTF50 and contrast per method ──
    fig, axes = plt.subplots(2, len(METHODS), figsize=(5 * len(METHODS), 9))

    for col, m in enumerate(METHODS):
        for row, (grid, title, cmap) in enumerate([
            (mtf50_grids[m],    f'{m}\nMTF50 (cy/mm)',   'viridis'),
            (contrast_grids[m], f'{m}\nMean contrast',   'plasma'),
        ]):
            ax = axes[row, col]
            vmax = np.nanmax(grid) if not np.all(np.isnan(grid)) else 1
            im = ax.imshow(grid, aspect='auto', cmap=cmap, vmin=0, vmax=vmax)
            ax.set_xticks(range(len(settles))); ax.set_xticklabels(settle_labels, rotation=40, ha='right', fontsize=7)
            ax.set_yticks(range(len(tilts)));   ax.set_yticklabels(tilt_labels, fontsize=7)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel('Settle time', fontsize=8)
            ax.set_ylabel('Tilt (rad)',  fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
            for ti in range(len(tilts)):
                for si in range(len(settles)):
                    v = grid[ti, si]
                    if not np.isnan(v):
                        fmt = f'{v:.0f}' if row == 0 else f'{v:.2f}'
                        ax.text(si, ti, fmt, ha='center', va='center', fontsize=6, color='white')

    plt.suptitle('SR Sweep: MTF50 and Ring Contrast\n(rows=tilt angle, cols=settle time)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'summary_heatmaps.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Improvement ratio: method MTF50 / Native MTF50 ──
    fig, axes = plt.subplots(1, len(METHODS) - 1, figsize=(5 * (len(METHODS) - 1), 4))
    native_grid = mtf50_grids['Native']
    for col, m in enumerate([x for x in METHODS if x != 'Native']):
        ax = axes[col]
        ratio = mtf50_grids[m] / native_grid
        vmax = np.nanmax(ratio) if not np.all(np.isnan(ratio)) else 2
        im = ax.imshow(ratio, aspect='auto', cmap='RdYlGn', vmin=0.8, vmax=vmax)
        ax.set_xticks(range(len(settles))); ax.set_xticklabels(settle_labels, rotation=40, ha='right', fontsize=7)
        ax.set_yticks(range(len(tilts)));   ax.set_yticklabels(tilt_labels, fontsize=7)
        ax.set_title(f'{m} / Native MTF50', fontsize=9)
        ax.set_xlabel('Settle time', fontsize=8)
        ax.set_ylabel('Tilt (rad)',  fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for ti in range(len(tilts)):
            for si in range(len(settles)):
                v = ratio[ti, si]
                if not np.isnan(v):
                    ax.text(si, ti, f'{v:.2f}×', ha='center', va='center', fontsize=6, color='black')

    plt.suptitle('MTF50 Improvement Ratio vs Native Bicubic', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'summary_improvement.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Line plots: MTF50 vs tilt for each settle time, per method ──
    fig, axes = plt.subplots(1, len(METHODS), figsize=(5 * len(METHODS), 4), sharey=True)
    cmap_lines = plt.cm.viridis(np.linspace(0, 1, len(settles)))
    for col, m in enumerate(METHODS):
        ax = axes[col]
        for si, settle in enumerate(settles):
            vals = mtf50_grids[m][:, si]
            ax.plot(tilts, vals, 'o-', color=cmap_lines[si], lw=1.5,
                    markersize=5, label=f'{settle:.0f} ms')
        ax.set_title(m, fontsize=10)
        ax.set_xlabel('Tilt angle (rad)', fontsize=8)
        ax.set_ylabel('MTF50 (cy/mm)', fontsize=8)
        ax.grid(True, alpha=0.3)
        if col == len(METHODS) - 1:
            ax.legend(fontsize=7, title='Settle time')

    plt.suptitle('MTF50 vs Tilt Angle (per settle time)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'summary_tilt_curves.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Print ranked table ──
    print('\n' + '─' * 70)
    print(f'{"Combination":<32} {"Native":>8} {"SAA":>8} {"IBP":>8} {"TV-SR":>8}')
    print('─' * 70)
    rows = []
    for key, metrics in all_results.items():
        tilt_str, settle_str = key.split('|')
        label = f'tilt {tilt_str}  settle {settle_str} ms'
        vals = [metrics['mtf'].get(m, {}).get('mtf50_cymm', float('nan')) for m in METHODS]
        rows.append((label, vals))

    # Sort by IBP MTF50 descending
    rows.sort(key=lambda x: x[1][METHODS.index('IBP')] if not np.isnan(x[1][METHODS.index('IBP')]) else -1, reverse=True)
    for label, vals in rows:
        val_str = ''.join(f'{v:8.1f}' if not np.isnan(v) else '     N/A' for v in vals)
        print(f'{label:<32}{val_str}')
    print('─' * 70)
    print('(MTF50 in cy/mm — higher is better)')

    # Save full summary JSON
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as fp:
        json.dump(all_results, fp, indent=2)
    print(f'\nResults saved to: {RESULTS_DIR}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('Loading PSF kernel ...')
    psf_hr = load_psf()
    print(f'  HR PSF kernel shape: {psf_hr.shape}')

    # Discover all tilt/settle combination folders
    combos = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('tilt')
    ])
    print(f'Found {len(combos)} combinations in {DATA_DIR}\n')
    print(f'LR crop: rows [{CROP_R0},{CROP_R1}], cols [{CROP_C0},{CROP_C1}]  '
          f'→ {CROP_R1-CROP_R0}×{CROP_C1-CROP_C0} px  '
          f'(HR: {(CROP_R1-CROP_R0)*f}×{(CROP_C1-CROP_C0)*f} px)\n')

    all_results = {}
    t_total = time.time()

    for i, combo in enumerate(combos):
        combo_dir = os.path.join(DATA_DIR, combo)
        out_dir   = os.path.join(RESULTS_DIR, combo)

        # Parse tilt and settle time from folder name: tilt0.11391_settle2.0ms
        parts      = combo.replace('ms', '').split('_settle')
        tilt_val   = float(parts[0].replace('tilt', ''))
        settle_val = float(parts[1])
        key        = f'{tilt_val}|{settle_val}'

        elapsed = time.time() - t_total
        eta_str = ''
        if i > 0:
            eta = elapsed / i * (len(combos) - i)
            eta_str = f'  ETA {eta/60:.1f} min'

        print(f'[{i+1:2d}/{len(combos)}] {combo}{eta_str}')
        metrics = process_combination(combo_dir, psf_hr, out_dir)
        all_results[key] = metrics

    print(f'\nAll combinations done in {(time.time()-t_total)/60:.1f} min')
    print('Generating summary ...')
    generate_summary(all_results)


if __name__ == '__main__':
    main()
