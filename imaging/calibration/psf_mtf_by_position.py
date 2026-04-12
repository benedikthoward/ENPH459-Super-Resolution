"""
PSF & MTF Characterisation Grouped by Shift Position
=====================================================
Processes the 20260325_184308 sweep dataset.  Images are grouped by the
9 mirror-position labels (pos0 … pos8) where pos4 = (0,0) shift.
For every position group the full PSF / MTF pipeline is run and the
results are saved together so positions can be compared.

Usage:
    python psf_mtf_by_position.py <folder>
        [--pixel-pitch-um PITCH]
        [--crop-radius PIXELS]
        [--psf-zoom PIXELS]
        [--output-dir DIR]

Example:
    python psf_mtf_by_position.py imaging/calibration/20260325_184308 \\
        --pixel-pitch-um 3.45 --output-dir imaging/calibration/20260325_184308
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.ndimage import center_of_mass, gaussian_filter, shift as ndi_shift
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Helpers (identical to psf_mtf.py)
# ---------------------------------------------------------------------------

def load_gray(path):
    img = np.array(Image.open(path), dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img


def find_peak(img, smooth_sigma=2.0):
    smoothed = gaussian_filter(img, sigma=smooth_sigma)
    return np.unravel_index(smoothed.argmax(), smoothed.shape)


def extract_psf(img, center, radius, bg_percentile=50.0):
    r, c = center
    h, w = img.shape
    r0, r1 = max(r - radius, 0), min(r + radius + 1, h)
    c0, c1 = max(c - radius, 0), min(c + radius + 1, w)
    roi = img[r0:r1, c0:c1].copy()

    mask = np.ones_like(roi, dtype=bool)
    cy, cx = roi.shape[0] // 2, roi.shape[1] // 2
    Y, X = np.ogrid[:roi.shape[0], :roi.shape[1]]
    inner = ((Y - cy)**2 + (X - cx)**2) < (radius * 0.6)**2
    mask[inner] = False
    bg = np.percentile(roi[mask], bg_percentile)

    roi -= bg
    roi[roi < 0] = 0

    # Threshold sparse noise floor — handles integer-valued cameras where
    # median bg = 0 but isolated 1-count pixels remain after subtraction.
    bg_std = np.std(roi[mask])
    if bg_std > 0:
        roi[roi < 3.0 * bg_std] = 0

    return roi


def subpixel_centre(psf):
    thresh = psf.max() * 0.1
    masked = np.where(psf > thresh, psf, 0)
    return center_of_mass(masked)


def radial_average(data_2d, center=None, max_radius=None):
    h, w = data_2d.shape
    if center is None:
        cy, cx = h / 2.0, w / 2.0
    else:
        cy, cx = center

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
    a = np.cos(theta)**2 / (2 * sigma_x**2) + np.sin(theta)**2 / (2 * sigma_y**2)
    b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
    c = np.sin(theta)**2 / (2 * sigma_x**2) + np.cos(theta)**2 / (2 * sigma_y**2)
    g = offset + amp * np.exp(
        -(a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2))
    return g.ravel()


def fit_gaussian_psf(psf):
    h, w = psf.shape
    y, x = np.mgrid[:h, :w]
    cy, cx = subpixel_centre(psf)

    p0 = [psf.max(), cx, cy, 2.0, 2.0, 0.0, 0.0]
    bounds_lo = [0, 0, 0, 0.3, 0.3, -np.pi, -np.inf]
    bounds_hi = [psf.max() * 2, w, h, w / 2, h / 2, np.pi, psf.max() * 0.5]

    try:
        popt, _ = curve_fit(gauss2d, (x, y), psf.ravel(), p0=p0,
                            bounds=(bounds_lo, bounds_hi), maxfev=20000)
        fit_img = gauss2d((x, y), *popt).reshape(h, w)
        return popt, fit_img
    except RuntimeError:
        return None, None


def compute_mtf(psf, pixel_pitch_um=None):
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
        freq_label = "cycles/mm"
        nyquist = 1.0 / (2.0 * pixel_pitch_um * 1e-3)
    else:
        freq = freq_cpp
        freq_label = "cycles/pixel"
        nyquist = 0.5

    return freq, mtf_profile, mtf_2d, freq_label, nyquist


def mtf_at_fraction(freq, mtf, fraction=0.5):
    above = mtf >= fraction
    if not above.any() or above.all():
        return np.nan
    idx = np.where(np.diff(above.astype(int)) == -1)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    f0, f1 = freq[i], freq[i + 1]
    m0, m1 = mtf[i], mtf[i + 1]
    if abs(m1 - m0) < 1e-12:
        return f0
    return f0 + (fraction - m0) * (f1 - f0) / (m1 - m0)


def zoom_crop(img, center, half_width):
    cy, cx = int(round(center[0])), int(round(center[1]))
    h, w = img.shape
    r0 = max(cy - half_width, 0)
    r1 = min(cy + half_width + 1, h)
    c0 = max(cx - half_width, 0)
    c1 = min(cx + half_width + 1, w)
    return img[r0:r1, c0:c1]


# ---------------------------------------------------------------------------
# Per-position analysis
# ---------------------------------------------------------------------------

POSITION_LABELS = {
    0: "(-x,+y)",
    1: "(0,+y)",
    2: "(+x,+y)",
    3: "(-x, 0)",
    4: "(0, 0)",
    5: "(+x, 0)",
    6: "(-x,-y)",
    7: "(0,-y)",
    8: "(+x,-y)",
}


def analyse_position(paths, crop_radius, pixel_pitch_um, bg_percentile=50.0):
    """Run the full PSF/MTF pipeline on a list of image paths.

    Returns a dict with all results for this position group.
    """
    if not paths:
        return None

    # Extract and align PSFs
    raw_psfs = []
    for p in paths:
        img = load_gray(p)
        peak = find_peak(img)
        psf = extract_psf(img, peak, crop_radius, bg_percentile)
        raw_psfs.append(psf)

    # Naive stack: simple average WITHOUT sub-pixel alignment.
    # Its MTF degrades for positions with a spread of shifts (shows real blurring).
    psf_naive = np.array(raw_psfs).mean(axis=0)

    target_c = np.array([crop_radius, crop_radius], dtype=float)
    aligned_psfs = []
    for psf in raw_psfs:
        com = np.array(subpixel_centre(psf))
        s = target_c - com
        aligned_psfs.append(np.clip(ndi_shift(psf, s, order=3, mode='constant'), 0, None))

    psf_stack = np.array(aligned_psfs)
    n = psf_stack.shape[0]
    psf_avg = psf_stack.mean(axis=0)
    psf_std = psf_stack.std(axis=0) if n > 1 else np.zeros_like(psf_avg)

    # Radial PSF profiles
    all_radial_psf, all_mtf, all_ee = [], [], []
    freq_ref = None
    freq_label = "cycles/pixel"
    nyquist = 0.5

    for psf in aligned_psfs:
        com = subpixel_centre(psf)
        radii, prof = radial_average(psf, com, crop_radius)
        mx = prof.max()
        all_radial_psf.append(prof / mx if mx > 0 else prof)

        ee = np.cumsum(prof * radii * 2 * np.pi)
        ee_total = ee[-1] if ee[-1] > 0 else 1.0
        all_ee.append(ee / ee_total)

        freq, mtf_r, _, fl, nq = compute_mtf(psf, pixel_pitch_um)
        all_mtf.append(mtf_r)
        freq_ref = freq
        freq_label = fl
        nyquist = nq

    all_radial_psf = np.array(all_radial_psf)
    all_mtf = np.array(all_mtf)
    all_ee = np.array(all_ee)

    radial_mean = all_radial_psf.mean(axis=0)
    radial_std = all_radial_psf.std(axis=0) if n > 1 else np.zeros_like(radial_mean)
    mtf_mean = all_mtf.mean(axis=0)
    mtf_std = all_mtf.std(axis=0) if n > 1 else np.zeros_like(mtf_mean)
    ee_mean = all_ee.mean(axis=0)
    ee_std = all_ee.std(axis=0) if n > 1 else np.zeros_like(ee_mean)

    # Gaussian fit on average PSF
    popt, fit_img = fit_gaussian_psf(psf_avg)

    # Per-image Gaussian fits
    all_sx, all_sy = [], []
    for psf in aligned_psfs:
        po, _ = fit_gaussian_psf(psf)
        if po is not None:
            all_sx.append(po[3])
            all_sy.append(po[4])

    # MTF metrics
    freq_avg, mtf_avg_r, mtf_2d_avg, _, _ = compute_mtf(psf_avg, pixel_pitch_um)
    mtf50 = mtf_at_fraction(freq_avg, mtf_avg_r, 0.5)
    mtf10 = mtf_at_fraction(freq_avg, mtf_avg_r, 0.1)

    all_mtf50 = np.array([mtf_at_fraction(freq_ref, m, 0.5) for m in all_mtf])
    all_mtf10 = np.array([mtf_at_fraction(freq_ref, m, 0.1) for m in all_mtf])

    com_avg = subpixel_centre(psf_avg)
    radii_psf, psf_profile_avg = radial_average(psf_avg, com_avg, crop_radius)

    # Naive-stack MTF
    freq_naive, mtf_naive_r, _, _, _ = compute_mtf(psf_naive, pixel_pitch_um)
    mtf50_naive = mtf_at_fraction(freq_naive, mtf_naive_r, 0.5)
    mtf10_naive = mtf_at_fraction(freq_naive, mtf_naive_r, 0.1)

    return dict(
        n_images=n,
        psf_avg=psf_avg,
        psf_std=psf_std,
        psf_naive=psf_naive,
        psf_fit=fit_img,
        psf_fit_params=np.array(popt) if popt is not None else None,
        com_avg=np.array(com_avg),
        radii_psf=radii_psf,
        psf_profile_avg=psf_profile_avg,
        radial_mean=radial_mean,
        radial_std=radial_std,
        freq=freq_ref,
        mtf_mean=mtf_mean,
        mtf_std=mtf_std,
        mtf_2d_avg=mtf_2d_avg,
        ee_mean=ee_mean,
        ee_std=ee_std,
        mtf50=mtf50,
        mtf10=mtf10,
        nyquist=nyquist,
        freq_label=freq_label,
        per_image_mtf50=all_mtf50,
        per_image_mtf10=all_mtf10,
        per_image_sigma_x=np.array(all_sx) if all_sx else np.array([]),
        per_image_sigma_y=np.array(all_sy) if all_sy else np.array([]),
        freq_naive=freq_naive,
        mtf_naive=mtf_naive_r,
        mtf50_naive=mtf50_naive,
        mtf10_naive=mtf10_naive,
        raw_psfs=raw_psfs,
    )


# ---------------------------------------------------------------------------
# Per-position figure (same style as psf_mtf.py)
# ---------------------------------------------------------------------------

def plot_position(res, pos_id, title_suffix="", zoom=10, output_path=None):
    n = res['n_images']
    psf_avg = res['psf_avg']
    psf_std = res['psf_std']
    com_avg = res['com_avg']
    radii_psf = res['radii_psf']
    radial_mean = res['radial_mean']
    radial_std = res['radial_std']
    freq_ref = res['freq']
    mtf_mean = res['mtf_mean']
    mtf_std = res['mtf_std']
    ee_mean = res['ee_mean']
    ee_std = res['ee_std']
    mtf_2d_avg = res['mtf_2d_avg']
    mtf50 = res['mtf50']
    mtf10 = res['mtf10']
    nyquist = res['nyquist']
    freq_label = res['freq_label']
    popt = res['psf_fit_params']
    fit_img = res['psf_fit']

    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    label = POSITION_LABELS.get(pos_id, "pos%d" % pos_id)
    fig.suptitle(
        "PSF & MTF – Position %d %s  (n=%d)  %s" % (pos_id, label, n, title_suffix),
        fontsize=13, fontweight="bold")
    gs = GridSpec(3, 3, figure=fig)

    # (0,0) Mean PSF linear
    ax = fig.add_subplot(gs[0, 0])
    psf_z = zoom_crop(psf_avg, com_avg, zoom)
    ax.imshow(psf_z, cmap='inferno', vmin=0, vmax=psf_avg.max(),
              origin='lower', interpolation='nearest',
              extent=[-zoom, zoom, -zoom, zoom])
    ax.set_title("Mean PSF (linear)")
    ax.set_xlabel("px"); ax.set_ylabel("px")

    # (0,1) Mean PSF log
    ax = fig.add_subplot(gs[0, 1])
    log_z = zoom_crop(np.log10(psf_avg + 1), com_avg, zoom)
    ax.imshow(log_z, cmap='inferno', origin='lower', interpolation='nearest',
              extent=[-zoom, zoom, -zoom, zoom])
    ax.set_title("Mean PSF (log10)")
    ax.set_xlabel("px"); ax.set_ylabel("px")

    # (0,2) PSF std
    ax = fig.add_subplot(gs[0, 2])
    if n > 1:
        std_z = zoom_crop(psf_std, com_avg, zoom)
        im = ax.imshow(std_z, cmap='magma', origin='lower', interpolation='nearest',
                       extent=[-zoom, zoom, -zoom, zoom])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("PSF std-dev (n=%d)" % n)
    else:
        ax.text(0.5, 0.5, "Single image", transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title("PSF std-dev")
    ax.set_xlabel("px"); ax.set_ylabel("px")

    # (1,0) Radial PSF
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(radii_psf, radial_mean, 'k-', lw=1.5, label='Mean')
    if n > 1:
        ax.fill_between(radii_psf,
                        np.clip(radial_mean - radial_std, 0, None),
                        radial_mean + radial_std,
                        alpha=0.25, color='steelblue', label='+/-1 std')
    if popt is not None:
        sx = popt[3]; sy = popt[4]
        sigma_avg = (sx + sy) / 2
        gauss_r = np.exp(-radii_psf**2 / (2 * sigma_avg**2))
        ax.plot(radii_psf, gauss_r, 'r--', lw=1, alpha=0.7,
                label='Gauss (σ=%.1f px)' % sigma_avg)
    ax.set_xlabel("Radius (px)")
    ax.set_ylabel("Normalised intensity")
    ax.set_title("Radial PSF")
    ax.legend(fontsize=7)
    ax.set_xlim(0, min(len(radii_psf) - 1, 25))
    ax.grid(True, alpha=0.3)

    # (1,1) Encircled energy
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(radii_psf, ee_mean, 'b-', lw=1.5, label='Mean')
    if n > 1:
        ax.fill_between(radii_psf,
                        np.clip(ee_mean - ee_std, 0, None),
                        np.clip(ee_mean + ee_std, None, 1.0),
                        alpha=0.25, color='steelblue', label='+/-1 std')
    for frac, ls in [(0.8, '--'), (0.9, ':')]:
        idx = np.searchsorted(ee_mean, frac)
        if idx < len(radii_psf):
            ax.axhline(frac, color='gray', ls=ls, alpha=0.5)
            ax.axvline(radii_psf[idx], color='gray', ls=ls, alpha=0.5)
            ax.annotate("EE%d @ r=%d px" % (int(frac * 100), radii_psf[idx]),
                        (radii_psf[idx], frac), fontsize=7,
                        xytext=(5, -10), textcoords='offset points')
    ax.set_xlabel("Radius (px)")
    ax.set_ylabel("Encircled energy")
    ax.set_title("Encircled Energy")
    ax.set_xlim(0, min(len(radii_psf) - 1, 25))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,2) PSF cross-sections
    ax = fig.add_subplot(gs[1, 2])
    cy_int, cx_int = int(round(com_avg[0])), int(round(com_avg[1]))
    h_slice = psf_avg[cy_int, :]
    v_slice = psf_avg[:, cx_int]
    hmax = max(h_slice.max(), v_slice.max(), 1)
    ax.plot(np.arange(len(h_slice)) - cx_int, h_slice / hmax, 'b-', lw=1, label='H')
    ax.plot(np.arange(len(v_slice)) - cy_int, v_slice / hmax, 'r-', lw=1, label='V')
    ax.set_xlabel("Offset (px)")
    ax.set_ylabel("Normalised intensity")
    ax.set_title("PSF cross-sections (mean)")
    ax.legend(fontsize=7)
    ax.set_xlim(-zoom, zoom)
    ax.grid(True, alpha=0.3)

    # (2,0) 2D MTF
    ax = fig.add_subplot(gs[2, 0])
    s2 = mtf_2d_avg.shape[0]; c2 = s2 // 2; r_show = int(s2 * 0.5)
    mtf_crop = mtf_2d_avg[c2 - r_show:c2 + r_show, c2 - r_show:c2 + r_show]
    ax.imshow(mtf_crop, cmap='viridis', vmin=0, vmax=1, origin='lower',
              extent=[-0.5, 0.5, -0.5, 0.5])
    ax.set_xlabel("fx (cy/px)")
    ax.set_ylabel("fy (cy/px)")
    ax.set_title("2D MTF (mean PSF)")

    # (2,1) Radial MTF
    ax = fig.add_subplot(gs[2, 1])
    valid = freq_ref <= nyquist
    ax.plot(freq_ref[valid], mtf_mean[valid], 'k-', lw=1.5, label='Aligned avg')
    if n > 1:
        ax.fill_between(freq_ref[valid],
                        np.clip(mtf_mean[valid] - mtf_std[valid], 0, None),
                        np.clip(mtf_mean[valid] + mtf_std[valid], None, 1.05),
                        alpha=0.25, color='steelblue', label='+/-1 std')
    # Naive-stack MTF: blurred by shift spread — meaningful difference across positions
    freq_n = res.get('freq_naive')
    mtf_n  = res.get('mtf_naive')
    mtf50_n = res.get('mtf50_naive', np.nan)
    if freq_n is not None and mtf_n is not None:
        valid_n = freq_n <= nyquist
        ax.plot(freq_n[valid_n], mtf_n[valid_n], 'g--', lw=1.2,
                label='Naive stack (MTF50=%.3f %s)' % (
                    mtf50_n if not np.isnan(mtf50_n) else 0, freq_label))
    ax.axhline(0.5, color='orange', ls='--', alpha=0.6,
               label='MTF50 = %.3f %s' % (mtf50, freq_label))
    ax.axhline(0.1, color='red', ls='--', alpha=0.6,
               label='MTF10 = %.3f %s' % (mtf10, freq_label))
    if not np.isnan(mtf50):
        ax.axvline(mtf50, color='orange', ls=':', alpha=0.4)
    if not np.isnan(mtf10):
        ax.axvline(mtf10, color='red', ls=':', alpha=0.4)
    ax.set_xlabel("Spatial frequency (%s)" % freq_label)
    ax.set_ylabel("MTF")
    ax.set_title("Radial MTF")
    ax.set_xlim(0, nyquist)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (2,2) Per-image MTF50 / MTF10 violin
    ax = fig.add_subplot(gs[2, 2])
    mtf50s = res['per_image_mtf50']
    mtf10s = res['per_image_mtf10']
    valid50 = mtf50s[~np.isnan(mtf50s)]
    valid10 = mtf10s[~np.isnan(mtf10s)]
    data_v = [d for d in [valid50, valid10] if len(d) > 1]
    labels_v = [lb for lb, d in
                zip(['MTF50', 'MTF10'], [valid50, valid10]) if len(d) > 1]
    if data_v:
        parts = ax.violinplot(data_v, showmedians=True)
        ax.set_xticks(range(1, len(labels_v) + 1))
        ax.set_xticklabels(labels_v)
    ax.set_ylabel("Spatial frequency (%s)" % freq_label)
    ax.set_title("Per-image MTF50/MTF10 distribution")
    ax.grid(True, alpha=0.3, axis='y')
    if len(valid50) > 0:
        ax.annotate("%.3f±%.3f" % (valid50.mean(), valid50.std()),
                    xy=(1, valid50.mean()), fontsize=7, ha='center', va='bottom')
    if len(valid10) > 0:
        ax.annotate("%.3f±%.3f" % (valid10.mean(), valid10.std()),
                    xy=(2, valid10.mean()), fontsize=7, ha='center', va='bottom')

    if output_path:
        plt.savefig(output_path, dpi=180)
        print("  Saved %s" % output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparison figure across all positions
# ---------------------------------------------------------------------------

def plot_comparison(results_by_pos, output_path=None):
    """
    Two-panel summary:
      - Left:  mean radial MTF for each position (all on one axes)
      - Right: MTF50 and MTF10 vs position index with error bars
    Plus a 3x3 grid of mean PSF thumbnails (one per position).
    """
    positions = sorted(results_by_pos.keys())
    n_pos = len(positions)

    # Color cycle
    cmap = plt.cm.get_cmap('tab10', n_pos)
    colors = {p: cmap(i) for i, p in enumerate(positions)}

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    fig.suptitle("PSF/MTF comparison across shift positions", fontsize=14, fontweight='bold')
    outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- Top-left: radial MTF overlay (naive stack — shows real blurring) ---
    ax_mtf = fig.add_subplot(outer[0, 0])
    for pos in positions:
        r = results_by_pos[pos]
        freq_n = r.get('freq_naive', r['freq'])
        mtf_n  = r.get('mtf_naive',  r['mtf_mean'])
        nyquist = r['nyquist']
        freq_label = r['freq_label']
        valid = freq_n <= nyquist
        label = "pos%d %s (n=%d)" % (pos, POSITION_LABELS.get(pos, ''), r['n_images'])
        ax_mtf.plot(freq_n[valid], mtf_n[valid],
                    color=colors[pos], lw=1.5 if pos == 4 else 0.9,
                    ls='-' if pos == 4 else '--', label=label)

    ax_mtf.axhline(0.5, color='gray', ls=':', alpha=0.5)
    ax_mtf.axhline(0.1, color='gray', ls=':', alpha=0.5)
    ax_mtf.set_xlabel("Spatial frequency (%s)" % freq_label)
    ax_mtf.set_ylabel("MTF")
    ax_mtf.set_title("Naive-stack MTF – all positions\n(unaligned avg; shows shift blurring)")
    ax_mtf.set_xlim(0, nyquist)
    ax_mtf.set_ylim(0, 1.05)
    ax_mtf.legend(fontsize=6, loc='upper right')
    ax_mtf.grid(True, alpha=0.3)

    # --- Top-right: MTF50 & MTF10 bar chart ---
    ax_bar = fig.add_subplot(outer[0, 1])
    pos_arr = np.array(positions)
    mtf50_means = []
    mtf50_stds = []
    mtf10_means = []
    mtf10_stds = []

    for pos in positions:
        r = results_by_pos[pos]
        # Use naive MTF50/MTF10 — reflects real blur across shift positions
        mtf50_means.append(r.get('mtf50_naive', r['mtf50']))
        mtf50_stds.append(0.0)
        mtf10_means.append(r.get('mtf10_naive', r['mtf10']))
        mtf10_stds.append(0.0)

    x = np.arange(n_pos)
    width = 0.35
    ax_bar.bar(x - width / 2, mtf50_means, width, yerr=mtf50_stds,
               label='MTF50', color='steelblue', capsize=3, alpha=0.8)
    ax_bar.bar(x + width / 2, mtf10_means, width, yerr=mtf10_stds,
               label='MTF10', color='salmon', capsize=3, alpha=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        ["p%d\n%s" % (p, POSITION_LABELS.get(p, '')) for p in positions],
        fontsize=7)
    ax_bar.set_ylabel("Spatial frequency (%s)" % freq_label)
    ax_bar.set_title("Naive-stack MTF50 / MTF10 by position")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, alpha=0.3, axis='y')

    # --- Bottom: 3x3 PSF thumbnail grid ---
    inner = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer[1, :],
                                             hspace=0.05, wspace=0.05)
    zoom = 15
    for i, pos in enumerate(range(9)):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(inner[row, col])
        if pos in results_by_pos:
            r = results_by_pos[pos]
            psf = r['psf_avg']
            com = r['com_avg']
            crop = zoom_crop(psf, com, zoom)
            ax.imshow(crop, cmap='inferno', vmin=0, vmax=psf.max(),
                      origin='lower', interpolation='nearest')
            ax.set_title("pos%d %s" % (pos, POSITION_LABELS.get(pos, '')),
                         fontsize=7)
        else:
            ax.text(0.5, 0.5, "pos%d\n(no data)" % pos,
                    transform=ax.transAxes, ha='center', va='center', fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    if output_path:
        plt.savefig(output_path, dpi=180)
        print("Saved comparison figure: %s" % output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Individual-image figure (for pos4 reference inspection)
# ---------------------------------------------------------------------------

def plot_individual_psfs(raw_psfs, pixel_pitch_um, n_show=6, zoom=15, output_path=None):
    """Show the first N raw (unaligned) PSFs side-by-side with their MTFs."""
    n_show = min(n_show, len(raw_psfs))
    if n_show == 0:
        return

    fig, axes = plt.subplots(2, n_show, figsize=(3.5 * n_show, 7),
                             constrained_layout=True)
    if n_show == 1:
        axes = np.array(axes).reshape(2, 1)
    fig.suptitle("Individual PSF images (pos4 / reference)", fontsize=12, fontweight='bold')

    for i, psf in enumerate(raw_psfs[:n_show]):
        com = subpixel_centre(psf)
        crop = zoom_crop(psf, com, zoom)

        # Top row: PSF image
        ax_img = axes[0, i]
        ax_img.imshow(crop, cmap='inferno', vmin=0, vmax=psf.max(),
                      origin='lower', interpolation='nearest',
                      extent=[-zoom, zoom, -zoom, zoom])
        ax_img.set_title("Image %d" % (i + 1), fontsize=9)
        ax_img.set_xlabel("px"); ax_img.set_ylabel("px")

        # Bottom row: radial MTF
        ax_mtf = axes[1, i]
        freq, mtf_r, _, freq_label, nyquist = compute_mtf(psf, pixel_pitch_um)
        valid = freq <= nyquist
        ax_mtf.plot(freq[valid], mtf_r[valid], 'k-', lw=1.2)
        mtf50 = mtf_at_fraction(freq, mtf_r, 0.5)
        ax_mtf.axhline(0.5, color='orange', ls='--', alpha=0.6,
                       label='MTF50=%.3f' % (mtf50 if not np.isnan(mtf50) else 0))
        ax_mtf.set_xlim(0, nyquist)
        ax_mtf.set_ylim(0, 1.05)
        ax_mtf.set_xlabel("Freq (%s)" % freq_label, fontsize=7)
        ax_mtf.set_ylabel("MTF", fontsize=7)
        ax_mtf.legend(fontsize=7)
        ax_mtf.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=180)
        print("  Saved %s" % output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_filename(path):
    """Parse image path for both dataset layouts.

    Flat format  (20260325_184308):
        sweepx_tilt0.02000_rep00_pos4.png

    Subfolder format  (20260326_123627):
        sweepx_tilt0.02000deg/pos4_(0,0).png   (repeat always 0)

    Returns dict with sweep_axis, tilt_angle, repeat, position, path,
    or None if the path does not match either format.
    """
    name    = os.path.basename(path)
    dirname = os.path.basename(os.path.dirname(path))

    # --- Flat format ---
    m = re.match(
        r'sweep([xy])_tilt([\d.]+)_rep(\d+)_pos(\d+)\.(png|tif|tiff|bmp|jpg|jpeg)$',
        name, re.IGNORECASE)
    if m:
        return dict(sweep_axis=m.group(1), tilt_angle=float(m.group(2)),
                    repeat=int(m.group(3)), position=int(m.group(4)), path=path)

    # --- Subfolder format ---
    m1 = re.match(r'sweep([xy])_tilt([\d.]+)deg$', dirname, re.IGNORECASE)
    m2 = re.match(r'pos(\d+)[_(].*\.(png|tif|tiff|bmp|jpg|jpeg)$', name, re.IGNORECASE)
    if m1 and m2:
        return dict(sweep_axis=m1.group(1), tilt_angle=float(m1.group(2)),
                    repeat=0, position=int(m2.group(1)), path=path)

    return None


def main():
    parser = argparse.ArgumentParser(description="PSF/MTF grouped by shift position")
    parser.add_argument("folder", help="Folder containing sweep images")
    parser.add_argument("--pixel-pitch-um", type=float, default=None)
    parser.add_argument("--crop-radius", type=int, default=50,
                        help="Half-width of ROI around PSF peak (pixels)")
    parser.add_argument("--psf-zoom", type=int, default=12,
                        help="Half-width in pixels for zoomed PSF display")
    parser.add_argument("--bg-percentile", type=float, default=50.0)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save outputs (default: same as folder)")
    args = parser.parse_args()

    folder = args.folder
    out_dir = args.output_dir if args.output_dir else folder
    os.makedirs(out_dir, exist_ok=True)

    # Gather all matching images — support both flat and one-level-deep subfolder layouts
    exts = ('*.png', '*.tif', '*.tiff', '*.bmp', '*.jpg', '*.jpeg')
    all_paths = []
    for ext in exts:
        all_paths.extend(glob.glob(os.path.join(folder, ext)))           # flat
        all_paths.extend(glob.glob(os.path.join(folder, '*', ext)))      # subfoldered
    all_paths = sorted(set(all_paths))

    records = [parse_filename(p) for p in all_paths]
    records = [r for r in records if r is not None]

    if not records:
        sys.exit("No matching sweep images found in: %s" % folder)

    # Group by position
    positions_found = sorted(set(r['position'] for r in records))
    print("Found %d images across positions: %s" % (len(records), positions_found))

    by_position = {pos: [] for pos in positions_found}
    for r in records:
        by_position[r['position']].append(r['path'])

    for pos in positions_found:
        n = len(by_position[pos])
        label = POSITION_LABELS.get(pos, "pos%d" % pos)
        print("  pos%d %s: %d images" % (pos, label, n))

    # Analyse each position
    results_by_pos = {}
    for pos in positions_found:
        label = POSITION_LABELS.get(pos, "pos%d" % pos)
        tag = "pos4 (0,0) – REFERENCE" if pos == 4 else "pos%d %s" % (pos, label)
        print("\n--- Analysing %s ---" % tag)
        paths = by_position[pos]
        res = analyse_position(paths, args.crop_radius, args.pixel_pitch_um,
                               args.bg_percentile)
        if res is None:
            print("  SKIP (no images)")
            continue
        results_by_pos[pos] = res

        sx_arr = res['per_image_sigma_x']
        sy_arr = res['per_image_sigma_y']
        if len(sx_arr) > 0:
            print("  sigma_x: %.2f +/- %.3f px  sigma_y: %.2f +/- %.3f px  (n=%d)" % (
                sx_arr.mean(), sx_arr.std(), sy_arr.mean(), sy_arr.std(), len(sx_arr)))
        v50 = res['per_image_mtf50']
        v50 = v50[~np.isnan(v50)]
        if len(v50) > 0:
            print("  MTF50: %.4f +/- %.4f %s" % (
                v50.mean(), v50.std(), res['freq_label']))

        fig_out = os.path.join(out_dir, "psf_mtf_pos%d.png" % pos)
        plot_position(res, pos, zoom=args.psf_zoom, output_path=fig_out)

        # For the reference position, also show individual images
        if pos == 4 and len(res.get('raw_psfs', [])) > 0:
            ind_out = os.path.join(out_dir, "psf_individual_pos4.png")
            plot_individual_psfs(res['raw_psfs'], args.pixel_pitch_um,
                                 n_show=min(6, len(res['raw_psfs'])),
                                 zoom=args.psf_zoom, output_path=ind_out)

    # Comparison figure
    comp_out = os.path.join(out_dir, "psf_mtf_comparison.png")
    plot_comparison(results_by_pos, output_path=comp_out)

    # Save numerical data
    save_dict = {}
    skip_keys = {'raw_psfs'}  # lists of arrays — not suitable for npz
    for pos, res in results_by_pos.items():
        prefix = "pos%d_" % pos
        for k, v in res.items():
            if k in skip_keys or v is None:
                continue
            if isinstance(v, np.ndarray):
                save_dict[prefix + k] = v
            elif isinstance(v, str):
                save_dict[prefix + k] = np.array(v)
            else:
                try:
                    save_dict[prefix + k] = np.array(v)
                except Exception:
                    pass

    save_dict['positions'] = np.array(list(results_by_pos.keys()))
    save_dict['position_labels'] = np.array(
        [POSITION_LABELS.get(p, "pos%d" % p) for p in results_by_pos.keys()])

    npz_out = os.path.join(out_dir, "psf_mtf_by_position_data.npz")
    np.savez(npz_out, **save_dict)
    print("\nSaved all numerical data to %s" % npz_out)
    print("Done.")


if __name__ == "__main__":
    main()
