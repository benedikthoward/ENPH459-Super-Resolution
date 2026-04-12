#!/usr/bin/env python3
"""
sweep_sr_20260323.py

Sweep tilt/settle combinations in 20260323_202456, run all SR methods,
evaluate quality metrics, and produce per-combination and summary results.

PSF source: imaging/calibration/20260326_152815 — pos4_(0,0).png averaged
            across all tilt-angle folders (pinhole at unshifted position).

ROIs are taken from chart_rois.ipynb (updated coordinates for this dataset):
  - Slanted edge  (300:480, 1280:1450)  → MTF50 via slanted-edge algorithm
  - Horiz wedge   (690:740, 1210:1290)  → fine-end Michelson contrast (lines horizontal,
                                          converge LEFT, profile traversed vertically per col)
  - Vert wedge    (480:540, 1070:1105)  → fine-end Michelson contrast (lines vertical,
                                          converge DOWN, profile traversed horizontally per row)

All three ROIs are covered by a combined LR crop (rows 290–770, cols 990–1460)
so IBP/TV-SR only operate on ~480×470 px instead of the full 1536×2048 frame.

Sweep filter: only tilts in [TILT_MIN, TILT_MAX] and settle == SETTLE_MS are run.
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
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 100})

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(SCRIPT_DIR, '20260323_202456')
RESULTS_DIR   = os.path.join(DATA_DIR, 'results')
PSF_CALIB_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'calibration', '20260326_152815')

# ── Sweep filter ───────────────────────────────────────────────────────────────
# Only process combinations with tilt in [TILT_MIN, TILT_MAX] and settle == SETTLE_MS
TILT_MIN  = 0.12    # rad
TILT_MAX  = 0.16    # rad
SETTLE_MS = 10.0    # ms

# ── Sensor / optics ────────────────────────────────────────────────────────────

PIXEL_PITCH_UM  = 3.45
UPSAMPLE_FACTOR = 2
f               = UPSAMPLE_FACTOR
PITCH_LR_MM     = PIXEL_PITCH_UM * 1e-3
PITCH_HR_MM     = PITCH_LR_MM / f
NYQUIST_LR      = 0.5 / PITCH_LR_MM    # 144.9 cy/mm
NYQUIST_HR      = 0.5 / PITCH_HR_MM    # 289.9 cy/mm

NOMINAL_SHIFTS = [
    (-0.5, -0.5),   # shift_0
    (+0.5, -0.5),   # shift_1
    (-0.5, +0.5),   # shift_2
    (+0.5, +0.5),   # shift_3
]

# ── ROI definitions (LR pixel coordinates in full image) ──────────────────────
# All from chart_rois.ipynb

EDGE_ROI_LR    = (300, 480, 1280, 1450)   # slanted diagonal spoke
HWEDGE_ROI_LR  = (700, 730, 1210, 1500)   # horiz lines, fine end LEFT (col 1210)
VWEDGE_ROI_LR  = (200, 540, 1070, 1100)   # vert  lines, fine end BOTTOM (row 540)

# Combined bounding box covering all three ROIs (+ small margin)
CROP_R0, CROP_R1 = 190, 740
CROP_C0, CROP_C1 = 1060, 1510

# ROIs in crop-local LR coordinates
def _to_crop(roi):
    r0, r1, c0, c1 = roi
    return (r0 - CROP_R0, r1 - CROP_R0, c0 - CROP_C0, c1 - CROP_C0)

EDGE_ROI_CROP   = _to_crop(EDGE_ROI_LR)
HWEDGE_ROI_CROP = _to_crop(HWEDGE_ROI_LR)
VWEDGE_ROI_CROP = _to_crop(VWEDGE_ROI_LR)

# Fraction of the wedge profile (from the fine end) used for the contrast metric
FINE_FRACTION = 0.30

# ── Iterative method parameters ───────────────────────────────────────────────

PSF_HALFWIDTH  = 3
IBP_ITERATIONS = 20
IBP_STEP_SIZE  = 0.5
TV_ITERATIONS  = 20
TV_STEP_SIZE   = 0.15
TV_LAMBDA      = 0.005

COLORS  = {'Native': 'C0', 'SAA': 'C2', 'IBP': 'C3', 'TV-SR': 'C4'}
METHODS = ['Native', 'SAA', 'IBP', 'TV-SR']

# ── Core image functions ───────────────────────────────────────────────────────

def load_gray(path):
    img = np.array(Image.open(path), dtype=np.float64)
    return img.mean(axis=2) if img.ndim == 3 else img


def blur(img, kernel):
    return fftconvolve(img, kernel, mode='same')


def forward_model(hr, kernel, shift_yx, factor):
    blurred = blur(hr, kernel)
    shifted = ndi_shift(blurred,
                        (shift_yx[0] * factor, shift_yx[1] * factor),
                        order=3, mode='nearest')
    return shifted[::factor, ::factor]


def back_project(error_lr, kernel, shift_yx, factor, hr_shape):
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
    n  = len(lr_list)
    errors = []
    for _ in range(n_iter):
        correction = np.zeros_like(hr)
        total_err  = 0.0
        for lr, s in zip(lr_list, shifts_yx):
            sim = forward_model(hr, kernel, s, factor)
            mh = min(sim.shape[0], lr.shape[0])
            mw = min(sim.shape[1], lr.shape[1])
            err = lr[:mh, :mw] - sim[:mh, :mw]
            total_err += np.mean(err ** 2)
            correction += back_project(err, kernel, s, factor, hr.shape)
        hr += step * correction / n
        hr  = np.clip(hr, 0, 255)
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
    n  = len(lr_list)
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
        hr  = np.clip(hr, 0, 255)
        errors.append(total_err / n)
    return hr, errors


# ── Metric functions ───────────────────────────────────────────────────────────

def slanted_line_mtf(img, roi, orientation='slanted', oversample=4):
    """
    MTF from a thin dark line on a bright background (NOT a step edge).

    The ISO slanted-edge algorithm differentiates an ESF to get the LSF, which
    only works for a half-field step edge.  For a thin dark line the intensity
    profile is bright→dark→bright (a bump), so the LSF IS the profile directly
    (inverted so the dark line becomes a positive peak).  We build an oversampled
    LSF by accumulating pixel intensities as a function of perpendicular distance
    from the fitted line centre, then FFT to get the MTF.

    Returns (freq_cycpx, mtf) where freq is in cycles/pixel of img.
    """
    r0, r1, c0, c1 = roi
    patch = img[r0:r1, c0:c1].copy()
    if orientation == 'horizontal':
        patch = patch.T
    rows, cols = patch.shape

    # Pass 1: find the line centre per row as the argmin of intensity
    # (the darkest column = centre of the dark line).
    line_pts = []
    for r in range(rows):
        row = patch[r, :].astype(float)
        if row.min() > 200:          # skip rows that are all-white
            continue
        line_pts.append((r, float(np.argmin(row))))
    if len(line_pts) < 10:
        return None, None
    line_pts = np.array(line_pts)

    # Pass 2: fit a line, keep inliers within 5 px of the fit.
    slope, intercept = np.polyfit(line_pts[:, 0], line_pts[:, 1], 1)
    predicted = slope * line_pts[:, 0] + intercept
    inliers   = np.abs(line_pts[:, 1] - predicted) < 5.0
    if inliers.sum() < 10:
        return None, None
    line_pts = line_pts[inliers]
    slope, intercept = np.polyfit(line_pts[:, 0], line_pts[:, 1], 1)
    cos_a = np.cos(np.arctan(slope))

    # Build oversampled LSF: accumulate intensities by perpendicular distance
    # from the fitted line centre.
    dists, vals = [], []
    for r in range(rows):
        line_c = slope * r + intercept
        for c in range(cols):
            dists.append((c - line_c) * cos_a)
            vals.append(patch[r, c])

    dists = np.array(dists)
    vals  = np.array(vals)
    bw    = 1.0 / oversample
    bins  = np.arange(dists.min(), dists.max(), bw)
    lsf_raw = np.zeros(len(bins))
    cnt     = np.zeros(len(bins))
    idx     = np.clip(((dists - dists.min()) / bw).astype(int), 0, len(bins) - 1)
    np.add.at(lsf_raw, idx, vals)
    np.add.at(cnt,     idx, 1)
    valid = cnt > 0
    lsf_raw[valid]  /= cnt[valid]
    lsf_raw[~valid]  = np.interp(bins[~valid], bins[valid], lsf_raw[valid])

    # Smooth lightly, invert (dark line → positive peak), normalise to unit area.
    lsf = gaussian_filter(lsf_raw, sigma=0.5)
    lsf = lsf.max() - lsf               # invert: dark peak becomes positive
    lsf_sum = lsf.sum()
    if lsf_sum < 1e-10:
        return None, None
    lsf /= lsf_sum                       # unit area → MTF(0) = 1 by construction

    lsf_w = lsf * np.hanning(len(lsf))
    dc_unwin = float(np.abs(np.fft.rfft(lsf))[0])
    mtf  = np.abs(np.fft.rfft(lsf_w))
    if dc_unwin > 1e-12:
        mtf /= dc_unwin
    freq = np.fft.rfftfreq(len(lsf_w), d=bw)    # cycles / pixel
    return freq, mtf


def compute_mtf50(freq_cymm, mtf):
    if freq_cymm is None or len(freq_cymm) < 2:
        return float('nan')
    for i in range(len(mtf) - 1):
        if mtf[i] >= 0.5 >= mtf[i + 1]:
            t = (0.5 - mtf[i]) / (mtf[i + 1] - mtf[i])
            return float(freq_cymm[i] + t * (freq_cymm[i + 1] - freq_cymm[i]))
    return float('nan')


def michelson_profile(profile_1d):
    """Michelson contrast of a 1-D intensity profile."""
    hi, lo = profile_1d.max(), profile_1d.min()
    denom = hi + lo
    if denom < 1e-6:
        return 0.0
    return float((hi - lo) / denom)


def _trim_bright_border(means_1d, threshold_frac=0.98):
    """
    Return (lo, hi) slice bounds that exclude leading/trailing elements whose
    mean ≥ threshold_frac * max.  Only trims contiguous bright regions at the
    edges — interior bright columns (white lines) are kept.
    """
    threshold = means_1d.max() * threshold_frac
    lo = 0
    while lo < len(means_1d) and means_1d[lo] >= threshold:
        lo += 1
    hi = len(means_1d) - 1
    while hi > lo and means_1d[hi] >= threshold:
        hi -= 1
    return lo, hi + 1   # hi+1 for Python slice


def wedge_contrast(img_crop, axis, fine_fraction=FINE_FRACTION, smooth=1):
    """
    Compute per-position Michelson contrast across a converging-line wedge crop.

    axis='h': lines are horizontal → profile along columns (vertical slice per col).
              fine end is col 0 (LEFT).
    axis='v': lines are vertical   → profile along rows (horizontal slice per row).
              fine end is last row (BOTTOM).

    White border regions (pure-white rows above/below for H-wedge, bright cols
    left/right for V-wedge) are automatically trimmed before computing contrast
    so they don't inflate the max of each profile.

    Returns:
        positions  : 1-D array of position indices
        contrasts  : Michelson contrast at each position (raw)
        fine_mean  : mean contrast over the fine-end `fine_fraction` of the crop
        coarse_mean: mean contrast over the coarse-end `fine_fraction` of the crop
    """
    if axis == 'h':
        # Trim rows that are white border (top/bottom) before taking col profiles
        row_means = img_crop.mean(axis=1)
        r0, r1 = _trim_bright_border(row_means)
        active_crop = img_crop[r0:r1, :]
        n = active_crop.shape[1]
        contrasts = np.array([michelson_profile(active_crop[:, c]) for c in range(n)])
    else:
        # Trim cols that are white border (left/right) before taking row profiles
        col_means = img_crop.mean(axis=0)
        c0, c1 = _trim_bright_border(col_means)
        active_crop = img_crop[:, c0:c1]
        n = active_crop.shape[0]
        contrasts = np.array([michelson_profile(active_crop[r, :]) for r in range(n)])

    contrasts_s = gaussian_filter1d(contrasts, sigma=smooth)
    k = max(1, int(n * fine_fraction))

    if axis == 'h':
        # fine end is index 0 (leftmost column = smallest spacing)
        fine_mean   = float(np.mean(contrasts_s[:k]))
        coarse_mean = float(np.mean(contrasts_s[-k:]))
    else:
        # fine end is the last row (bottom = closest to chart centre)
        fine_mean   = float(np.mean(contrasts_s[-k:]))
        coarse_mean = float(np.mean(contrasts_s[:k]))

    return np.arange(n), contrasts, fine_mean, coarse_mean


# ── PSF loading ────────────────────────────────────────────────────────────────

def load_psf():
    """
    Build PSF kernel from pos4_(0,0).png images in PSF_CALIB_DIR.
    All tilt-angle sub-folders are averaged (PSF is tilt-independent),
    then a PSF_HALFWIDTH-radius crop is extracted, background-subtracted,
    and upsampled to HR pixel spacing.
    """
    pos4_imgs = []
    for sweep_dir in sorted(os.listdir(PSF_CALIB_DIR)):
        full = os.path.join(PSF_CALIB_DIR, sweep_dir)
        if not os.path.isdir(full):
            continue
        pos4_path = os.path.join(full, 'pos4_(0,0).png')
        if os.path.exists(pos4_path):
            img = np.array(Image.open(pos4_path), dtype=np.float64)
            if img.ndim == 3:
                img = img.mean(axis=2)
            pos4_imgs.append(img)

    if not pos4_imgs:
        raise FileNotFoundError(f'No pos4_(0,0).png found under {PSF_CALIB_DIR}')

    avg  = np.mean(pos4_imgs, axis=0)
    peak = np.unravel_index(avg.argmax(), avg.shape)
    pr, pc = peak

    R      = PSF_HALFWIDTH
    kernel = avg[pr - R:pr + R + 1, pc - R:pc + R + 1].copy()

    # Background subtract using the four 3×3 corners of the patch
    corners = np.concatenate([
        kernel[:3, :3].ravel(), kernel[:3, -3:].ravel(),
        kernel[-3:, :3].ravel(), kernel[-3:, -3:].ravel(),
    ])
    kernel -= np.mean(corners)
    kernel  = np.clip(kernel, 0, None)
    kernel /= kernel.sum()

    # Upsample to HR pixel pitch
    kernel_hr  = ndi_zoom(kernel, f, order=3)
    kernel_hr  = np.clip(kernel_hr, 0, None)
    kernel_hr /= kernel_hr.sum()
    print(f'  PSF: averaged {len(pos4_imgs)} pos4 images, kernel {kernel.shape} → HR {kernel_hr.shape}')
    return kernel_hr


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
    shifts_lr  = [load_gray(os.path.join(combo_dir, f'shift_{i}.png')) for i in range(4)]

    center_crop = center_lr[CROP_R0:CROP_R1, CROP_C0:CROP_C1]
    shifts_crop  = [s[CROP_R0:CROP_R1, CROP_C0:CROP_C1] for s in shifts_lr]

    # SR methods
    native_hr = ndi_zoom(center_crop, f, order=3)
    saa_hr    = shift_and_add(shifts_crop, NOMINAL_SHIFTS, f, order=3)

    print('    IBP  ...', end=' ', flush=True)
    t0 = time.time()
    ibp_hr, ibp_errors = ibp(
        shifts_crop, NOMINAL_SHIFTS, psf_hr, saa_hr.copy(),
        factor=f, n_iter=IBP_ITERATIONS, step=IBP_STEP_SIZE,
    )
    print(f'{time.time() - t0:.0f}s')

    print('    TV-SR...', end=' ', flush=True)
    t0 = time.time()
    tvsr_hr, tv_errors = tv_sr(
        shifts_crop, NOMINAL_SHIFTS, psf_hr, saa_hr.copy(),
        factor=f, n_iter=TV_ITERATIONS, step=TV_STEP_SIZE, lam=TV_LAMBDA,
    )
    print(f'{time.time() - t0:.0f}s')

    hr_images = {'Native': native_hr, 'SAA': saa_hr, 'IBP': ibp_hr, 'TV-SR': tvsr_hr}

    # ── Slanted-edge MTF ──
    roi_hr_edge = tuple(x * f for x in EDGE_ROI_CROP)
    mtf_metrics = {}
    for name, img in hr_images.items():
        freq, mtf = slanted_line_mtf(img, roi_hr_edge, orientation='slanted')
        freq_cymm = freq / PITCH_HR_MM if freq is not None else None
        mtf_metrics[name] = {'mtf50_cymm': compute_mtf50(freq_cymm, mtf)}

    # LR reference
    freq_lr, mtf_lr = slanted_line_mtf(center_crop, EDGE_ROI_CROP, orientation='slanted')
    mtf_metrics['Native_LR'] = {
        'mtf50_cymm': compute_mtf50(
            freq_lr / PITCH_LR_MM if freq_lr is not None else None, mtf_lr,
        )
    }

    # ── Wedge contrast metrics (HR) ──
    roi_hr_hw = tuple(x * f for x in HWEDGE_ROI_CROP)
    roi_hr_vw = tuple(x * f for x in VWEDGE_ROI_CROP)

    wedge_metrics = {}
    for name, img in hr_images.items():
        r0, r1, c0, c1 = roi_hr_hw
        _, _, h_fine, h_coarse = wedge_contrast(img[r0:r1, c0:c1], axis='h')
        r0, r1, c0, c1 = roi_hr_vw
        _, _, v_fine, v_coarse = wedge_contrast(img[r0:r1, c0:c1], axis='v')
        wedge_metrics[name] = {
            'horiz_fine_contrast':   h_fine,
            'horiz_coarse_contrast': h_coarse,
            'vert_fine_contrast':    v_fine,
            'vert_coarse_contrast':  v_coarse,
        }

    # LR wedge reference
    r0, r1, c0, c1 = HWEDGE_ROI_CROP
    _, _, h_fine_lr, h_coarse_lr = wedge_contrast(center_crop[r0:r1, c0:c1], axis='h')
    r0, r1, c0, c1 = VWEDGE_ROI_CROP
    _, _, v_fine_lr, v_coarse_lr = wedge_contrast(center_crop[r0:r1, c0:c1], axis='v')
    wedge_metrics['Native_LR'] = {
        'horiz_fine_contrast':   h_fine_lr,
        'horiz_coarse_contrast': h_coarse_lr,
        'vert_fine_contrast':    v_fine_lr,
        'vert_coarse_contrast':  v_coarse_lr,
    }

    metrics = {
        'mtf':          mtf_metrics,
        'wedge':        wedge_metrics,
        'ibp_final_mse': float(ibp_errors[-1]) if ibp_errors else float('nan'),
        'tv_final_mse':  float(tv_errors[-1])  if tv_errors  else float('nan'),
    }

    with open(metrics_path, 'w') as fp:
        json.dump(metrics, fp, indent=2)

    _save_comparison_plot(hr_images, out_dir)
    _save_mtf_plot(hr_images, roi_hr_edge, out_dir)
    _save_wedge_plot(hr_images, roi_hr_hw, roi_hr_vw, out_dir)
    _save_convergence_plot(ibp_errors, tv_errors, out_dir)

    return metrics


def _save_comparison_plot(hr_images, out_dir):
    """Side-by-side crop of all three ROIs for each method."""
    rois = [
        ('Slanted edge', tuple(x * f for x in EDGE_ROI_CROP)),
        ('Horiz wedge',  tuple(x * f for x in HWEDGE_ROI_CROP)),
        ('Vert wedge',   tuple(x * f for x in VWEDGE_ROI_CROP)),
    ]
    n_methods = len(hr_images)
    n_rois    = len(rois)
    fig, axes = plt.subplots(n_rois, n_methods,
                             figsize=(4 * n_methods, 3 * n_rois))
    for ri, (roi_name, (r0, r1, c0, c1)) in enumerate(rois):
        for mi, (mname, img) in enumerate(hr_images.items()):
            ax = axes[ri, mi]
            ax.imshow(img[r0:r1, c0:c1], cmap='gray', interpolation='nearest')
            ax.set_title(f'{mname}' if ri == 0 else '', fontsize=9)
            ax.set_ylabel(roi_name if mi == 0 else '', fontsize=9)
            ax.axis('off')
    plt.suptitle('Method comparison — all ROIs', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison.png'), bbox_inches='tight')
    plt.close(fig)


def _save_mtf_plot(hr_images, roi_hr_edge, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, img in hr_images.items():
        freq, mtf = slanted_line_mtf(img, roi_hr_edge, orientation='slanted')
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


def _save_wedge_plot(hr_images, roi_hr_hw, roi_hr_vw, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, img in hr_images.items():
        c = COLORS.get(name, 'gray')
        r0, r1, c0, c1 = roi_hr_hw
        _, contrasts_h, _, _ = wedge_contrast(img[r0:r1, c0:c1], axis='h')
        axes[0].plot(gaussian_filter1d(contrasts_h, 2), color=c, lw=1.5, label=name)

        r0, r1, c0, c1 = roi_hr_vw
        _, contrasts_v, _, _ = wedge_contrast(img[r0:r1, c0:c1], axis='v')
        axes[1].plot(gaussian_filter1d(contrasts_v, 2), color=c, lw=1.5, label=name)

    for ax, title, xlabel in [
        (axes[0], 'Horiz wedge contrast (col 0 = fine end)', 'Column in crop (0=fine)'),
        (axes[1], 'Vert wedge contrast  (right = fine end)', 'Row in crop (high=fine)'),
    ]:
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Michelson contrast', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.suptitle('Wedge contrast profiles', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'wedge_contrast.png'), bbox_inches='tight')
    plt.close(fig)


def _save_convergence_plot(ibp_errors, tv_errors, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ibp_errors, color=COLORS['IBP'], lw=1.5)
    axes[0].set_title('IBP convergence')
    axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('MSE')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(tv_errors, color=COLORS['TV-SR'], lw=1.5)
    axes[1].set_title('TV-SR convergence')
    axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('MSE')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'convergence.png'), bbox_inches='tight')
    plt.close(fig)


# ── Summary ────────────────────────────────────────────────────────────────────

def generate_summary(all_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tilts   = sorted(set(float(k.split('|')[0]) for k in all_results))
    settles = sorted(set(float(k.split('|')[1]) for k in all_results))

    settle_labels = [f'{s:.0f} ms' for s in settles]
    tilt_labels   = [f'{t:.5f}' for t in tilts]

    # Build grids: rows=tilt, cols=settle
    def make_grid(metric_fn):
        g = {m: np.full((len(tilts), len(settles)), np.nan) for m in METHODS}
        for key, metrics in all_results.items():
            ti = tilts.index(float(key.split('|')[0]))
            si = settles.index(float(key.split('|')[1]))
            for m in METHODS:
                v = metric_fn(metrics, m)
                if v is not None:
                    g[m][ti, si] = v
        return g

    mtf50_grid  = make_grid(lambda m, name: m.get('mtf',  {}).get(name, {}).get('mtf50_cymm'))
    hfine_grid  = make_grid(lambda m, name: m.get('wedge', {}).get(name, {}).get('horiz_fine_contrast'))
    vfine_grid  = make_grid(lambda m, name: m.get('wedge', {}).get(name, {}).get('vert_fine_contrast'))

    # ── Heatmaps: one figure per metric ──
    for grid, fname, metric_label, cmap in [
        (mtf50_grid,  'summary_mtf50.png',       'MTF50 (cy/mm)',             'viridis'),
        (hfine_grid,  'summary_horiz_contrast.png','Horiz wedge fine contrast', 'plasma'),
        (vfine_grid,  'summary_vert_contrast.png', 'Vert wedge fine contrast',  'plasma'),
    ]:
        fig, axes = plt.subplots(1, len(METHODS), figsize=(5 * len(METHODS), 4))
        for col, m in enumerate(METHODS):
            ax  = axes[col]
            g   = grid[m]
            vmax = np.nanmax(g) if not np.all(np.isnan(g)) else 1
            im  = ax.imshow(g, aspect='auto', cmap=cmap, vmin=0, vmax=vmax)
            ax.set_xticks(range(len(settles))); ax.set_xticklabels(settle_labels, rotation=40, ha='right', fontsize=7)
            ax.set_yticks(range(len(tilts)));   ax.set_yticklabels(tilt_labels, fontsize=7)
            ax.set_title(f'{m}\n{metric_label}', fontsize=9)
            ax.set_xlabel('Settle time', fontsize=8)
            ax.set_ylabel('Tilt (rad)',  fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
            for ti in range(len(tilts)):
                for si in range(len(settles)):
                    v = g[ti, si]
                    if not np.isnan(v):
                        ax.text(si, ti, f'{v:.2f}', ha='center', va='center', fontsize=5, color='white')
        plt.suptitle(f'SR Sweep: {metric_label}\n(rows=tilt, cols=settle)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ── Improvement ratio vs Native (MTF50) ──
    fig, axes = plt.subplots(1, len(METHODS) - 1, figsize=(5 * (len(METHODS) - 1), 4))
    native_mtf = mtf50_grid['Native']
    for col, m in enumerate([x for x in METHODS if x != 'Native']):
        ax    = axes[col]
        ratio = mtf50_grid[m] / native_mtf
        vmax  = np.nanmax(ratio) if not np.all(np.isnan(ratio)) else 2
        im    = ax.imshow(ratio, aspect='auto', cmap='RdYlGn', vmin=0.8, vmax=vmax)
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
                    ax.text(si, ti, f'{v:.2f}×', ha='center', va='center', fontsize=5, color='black')
    plt.suptitle('MTF50 Improvement Ratio vs Native Bicubic', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'summary_improvement.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── MTF50 vs tilt, per settle time ──
    fig, axes = plt.subplots(1, len(METHODS), figsize=(5 * len(METHODS), 4), sharey=True)
    cmap_lines = plt.cm.viridis(np.linspace(0, 1, len(settles)))
    for col, m in enumerate(METHODS):
        ax = axes[col]
        for si, settle in enumerate(settles):
            ax.plot(tilts, mtf50_grid[m][:, si], 'o-',
                    color=cmap_lines[si], lw=1.5, markersize=5, label=f'{settle:.0f} ms')
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

    # ── Wedge fine contrast vs tilt ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, grid, title in [
        (axes[0], hfine_grid, 'Horiz wedge fine-end contrast vs tilt'),
        (axes[1], vfine_grid,  'Vert wedge fine-end contrast vs tilt'),
    ]:
        for m in METHODS:
            mean_per_tilt = np.nanmean(grid[m], axis=1)  # avg over settle times
            ax.plot(tilts, mean_per_tilt, 'o-', color=COLORS[m], lw=1.5, label=m)
        ax.set_xlabel('Tilt angle (rad)', fontsize=9)
        ax.set_ylabel('Michelson contrast (fine end)', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Wedge fine-end contrast vs tilt (averaged over settle times)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'summary_wedge_vs_tilt.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Print ranked table ──
    print('\n' + '─' * 80)
    print(f'{"Combination":<36} {"Nat MTF50":>9} {"SAA":>9} {"IBP":>9} {"TV-SR":>9}')
    print('─' * 80)
    rows = []
    for key, metrics in all_results.items():
        tilt_str, settle_str = key.split('|')
        label = f'tilt {tilt_str}  settle {settle_str} ms'
        vals  = [metrics['mtf'].get(m, {}).get('mtf50_cymm', float('nan')) for m in METHODS]
        h_fine = [metrics.get('wedge', {}).get(m, {}).get('horiz_fine_contrast', float('nan')) for m in METHODS]
        rows.append((label, vals, h_fine))

    rows.sort(key=lambda x: x[1][METHODS.index('IBP')] if not np.isnan(x[1][METHODS.index('IBP')]) else -1, reverse=True)
    for label, vals, _ in rows:
        val_str = ''.join(f'{v:9.1f}' if not np.isnan(v) else '      N/A' for v in vals)
        print(f'{label:<36}{val_str}')
    print('─' * 80)
    print('(MTF50 in cy/mm — higher is better)\n')

    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as fp:
        json.dump(all_results, fp, indent=2)
    print(f'Results saved to: {RESULTS_DIR}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('Loading PSF kernel ...')
    psf_hr = load_psf()
    print(f'  HR PSF kernel shape: {psf_hr.shape}')

    all_combos = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('tilt')
    ])

    # Filter to the tilt/settle range of interest
    combos = []
    for d in all_combos:
        parts      = d.replace('ms', '').split('_settle')
        tilt_val   = float(parts[0].replace('tilt', ''))
        settle_val = float(parts[1])
        if TILT_MIN <= tilt_val <= TILT_MAX and settle_val == SETTLE_MS:
            combos.append(d)
    combos = sorted(combos)

    print(f'Found {len(all_combos)} total combinations; running {len(combos)} '
          f'(tilt {TILT_MIN}–{TILT_MAX} rad, settle {SETTLE_MS:.0f} ms)\n')
    print(f'LR crop: rows [{CROP_R0},{CROP_R1}], cols [{CROP_C0},{CROP_C1}]  '
          f'→ {CROP_R1-CROP_R0}×{CROP_C1-CROP_C0} px  '
          f'(HR: {(CROP_R1-CROP_R0)*f}×{(CROP_C1-CROP_C0)*f} px)\n')

    all_results = {}
    t_total = time.time()

    for i, combo in enumerate(combos):
        combo_dir = os.path.join(DATA_DIR, combo)
        out_dir   = os.path.join(RESULTS_DIR, combo)

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
