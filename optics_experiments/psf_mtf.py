"""
PSF & MTF Characterization from Backlit Aperture Images
========================================================
Processes a folder of backlit pinhole/aperture images to extract each PSF,
compute per-image MTFs, and plot the mean result with variation bands.

Usage:
    python psf_mtf.py <folder_or_glob> [--pixel-pitch-um PITCH] [--aperture-um SIZE]
                                        [--crop-radius PIXELS] [--psf-zoom PIXELS]

Example:
    python psf_mtf.py imaging/calibration/ --pixel-pitch-um 3.45 --aperture-um 5
    python psf_mtf.py "imaging/calibration/tilt0*.png" --pixel-pitch-um 3.45
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.ndimage import center_of_mass, gaussian_filter, shift as ndi_shift
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_gray(path):
    """Load image as float64 grayscale (average of RGB if colour)."""
    img = np.array(Image.open(path), dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img


def find_peak(img, smooth_sigma=2.0):
    """Return (row, col) of the PSF peak after light smoothing."""
    smoothed = gaussian_filter(img, sigma=smooth_sigma)
    return np.unravel_index(smoothed.argmax(), smoothed.shape)


def extract_psf(img, center, radius, bg_percentile=50.0):
    """Extract background-subtracted ROI centred on peak."""
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
    return roi


def subpixel_centre(psf):
    """Centre-of-mass on thresholded PSF for subpixel accuracy."""
    thresh = psf.max() * 0.1
    masked = np.where(psf > thresh, psf, 0)
    return center_of_mass(masked)


def radial_average(data_2d, center=None, max_radius=None):
    """Azimuthally-averaged radial profile. Returns (radii, profile)."""
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
    """2D Gaussian for PSF fitting."""
    x, y = xy
    a = np.cos(theta)**2 / (2 * sigma_x**2) + np.sin(theta)**2 / (2 * sigma_y**2)
    b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
    c = np.sin(theta)**2 / (2 * sigma_x**2) + np.cos(theta)**2 / (2 * sigma_y**2)
    g = offset + amp * np.exp(-(a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2))
    return g.ravel()


def fit_gaussian_psf(psf):
    """Fit 2D Gaussian. Returns (params, fit_image) or (None, None)."""
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
        print("WARNING: Gaussian fit did not converge.", file=sys.stderr)
        return None, None


def compute_mtf(psf, pixel_pitch_um=None):
    """
    MTF from PSF via 2D FFT + radial averaging.
    Returns (freq, mtf_radial, mtf_2d, freq_label, nyquist).
    """
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
    """Frequency where MTF first drops below `fraction`."""
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
    """Extract a zoomed square from img around center (row, col)."""
    cy, cx = int(round(center[0])), int(round(center[1]))
    h, w = img.shape
    r0 = max(cy - half_width, 0)
    r1 = min(cy + half_width + 1, h)
    c0 = max(cx - half_width, 0)
    c1 = min(cx + half_width + 1, w)
    return img[r0:r1, c0:c1]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PSF / MTF characterisation")
    parser.add_argument("images", help="Folder path or glob pattern for images")
    parser.add_argument("--pixel-pitch-um", type=float, default=None,
                        help="Pixel pitch in micrometres (for physical freq axis)")
    parser.add_argument("--aperture-um", type=float, default=5.0,
                        help="Backlit aperture diameter in um (for reference)")
    parser.add_argument("--crop-radius", type=int, default=40,
                        help="Half-width of ROI to extract around peak (pixels)")
    parser.add_argument("--psf-zoom", type=int, default=10,
                        help="Half-width in pixels for zoomed PSF display")
    parser.add_argument("--bg-percentile", type=float, default=50.0,
                        help="Percentile for background estimation")
    parser.add_argument("--output", type=str, default="psf_mtf_results.png",
                        help="Output figure filename")
    args = parser.parse_args()

    # ---- Gather images ----
    target = args.images
    if os.path.isdir(target):
        exts = ('*.png', '*.tif', '*.tiff', '*.bmp', '*.jpg', '*.jpeg')
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(target, ext)))
        paths = sorted(paths)
    else:
        paths = sorted(glob.glob(target))

    if not paths:
        sys.exit("No image files found matching: %s" % target)
    print("Found %d image(s)" % len(paths))

    # ---- Extract and align all PSFs ----
    raw_psfs = []
    for p in paths:
        img = load_gray(p)
        peak = find_peak(img)
        psf = extract_psf(img, peak, args.crop_radius, args.bg_percentile)
        raw_psfs.append(psf)

    # Align each PSF so COM sits at ROI centre
    target_c = np.array([args.crop_radius, args.crop_radius], dtype=float)
    aligned_psfs = []
    for psf in raw_psfs:
        com = np.array(subpixel_centre(psf))
        s = target_c - com
        aligned_psfs.append(np.clip(ndi_shift(psf, s, order=3, mode='constant'), 0, None))

    psf_stack = np.array(aligned_psfs)  # (N, H, W)
    n_images = psf_stack.shape[0]
    psf_avg = psf_stack.mean(axis=0)
    psf_std = psf_stack.std(axis=0) if n_images > 1 else np.zeros_like(psf_avg)
    print("Aligned and stacked %d PSF(s)" % n_images)

    # ---- Per-image radial profiles and MTFs ----
    all_radial_psf = []
    all_mtf = []
    all_ee = []
    freq_ref = None
    freq_label = "cycles/pixel"
    nyquist = 0.5

    for psf in aligned_psfs:
        com = subpixel_centre(psf)
        radii, prof = radial_average(psf, com, args.crop_radius)
        mx = prof.max()
        all_radial_psf.append(prof / mx if mx > 0 else prof)

        # Encircled energy
        ee = np.cumsum(prof * radii * 2 * np.pi)
        ee_total = ee[-1] if ee[-1] > 0 else 1.0
        all_ee.append(ee / ee_total)

        # MTF
        freq, mtf_r, _, fl, nq = compute_mtf(psf, args.pixel_pitch_um)
        all_mtf.append(mtf_r)
        freq_ref = freq
        freq_label = fl
        nyquist = nq

    all_radial_psf = np.array(all_radial_psf)
    all_mtf = np.array(all_mtf)
    all_ee = np.array(all_ee)

    radial_mean = all_radial_psf.mean(axis=0)
    radial_std = all_radial_psf.std(axis=0) if n_images > 1 else np.zeros_like(radial_mean)
    mtf_mean = all_mtf.mean(axis=0)
    mtf_std = all_mtf.std(axis=0) if n_images > 1 else np.zeros_like(mtf_mean)
    ee_mean = all_ee.mean(axis=0)
    ee_std = all_ee.std(axis=0) if n_images > 1 else np.zeros_like(ee_mean)

    # ---- Gaussian fit on average PSF ----
    popt, fit_img = fit_gaussian_psf(psf_avg)
    sx, sy = None, None
    if popt is not None:
        amp, x0, y0, sx, sy, theta, off = popt
        fwhm_x = 2.355 * sx
        fwhm_y = 2.355 * sy
        print("Gaussian fit:  sigma_x=%.2f px, sigma_y=%.2f px" % (sx, sy))
        print("               FWHM_x=%.2f px, FWHM_y=%.2f px" % (fwhm_x, fwhm_y))
        if args.pixel_pitch_um:
            print("               FWHM_x=%.2f um, FWHM_y=%.2f um"
                  % (fwhm_x * args.pixel_pitch_um, fwhm_y * args.pixel_pitch_um))

    # ---- Per-image Gaussian fit stats ----
    all_sx, all_sy = [], []
    for psf in aligned_psfs:
        po, _ = fit_gaussian_psf(psf)
        if po is not None:
            all_sx.append(po[3])
            all_sy.append(po[4])
    if len(all_sx) > 1:
        print("Per-image sigma_x: %.2f +/- %.2f px  (n=%d)"
              % (np.mean(all_sx), np.std(all_sx), len(all_sx)))
        print("Per-image sigma_y: %.2f +/- %.2f px" % (np.mean(all_sy), np.std(all_sy)))

    # ---- MTF metrics on average ----
    freq_avg, mtf_avg_r, mtf_2d_avg, _, _ = compute_mtf(psf_avg, args.pixel_pitch_um)
    mtf50 = mtf_at_fraction(freq_avg, mtf_avg_r, 0.5)
    mtf10 = mtf_at_fraction(freq_avg, mtf_avg_r, 0.1)
    print("MTF50 = %.1f %s" % (mtf50, freq_label))
    print("MTF10 = %.1f %s" % (mtf10, freq_label))
    print("Nyquist = %.1f %s" % (nyquist, freq_label))

    # Per-image MTF50 / MTF10 for stats
    all_mtf50, all_mtf10 = [], []
    for m in all_mtf:
        all_mtf50.append(mtf_at_fraction(freq_ref, m, 0.5))
        all_mtf10.append(mtf_at_fraction(freq_ref, m, 0.1))
    all_mtf50 = np.array(all_mtf50)
    all_mtf10 = np.array(all_mtf10)
    if n_images > 1:
        valid50 = all_mtf50[~np.isnan(all_mtf50)]
        valid10 = all_mtf10[~np.isnan(all_mtf10)]
        if len(valid50) > 0:
            print("MTF50 across images: %.1f +/- %.1f %s"
                  % (np.mean(valid50), np.std(valid50), freq_label))
        if len(valid10) > 0:
            print("MTF10 across images: %.1f +/- %.1f %s"
                  % (np.mean(valid10), np.std(valid10), freq_label))

    # ---- Encircled energy on average ----
    com_avg = subpixel_centre(psf_avg)
    radii_psf, psf_profile_avg = radial_average(psf_avg, com_avg, args.crop_radius)

    # =====================================================================
    #  PLOT  (3 x 3 grid)
    # =====================================================================
    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    n_str = "(n=%d)" % n_images
    fig.suptitle("PSF & MTF Characterisation  %s" % n_str, fontsize=14, fontweight="bold")
    gs = GridSpec(3, 3, figure=fig)

    zoom = args.psf_zoom

    # --- (0,0) Mean PSF zoomed, linear ---
    ax1 = fig.add_subplot(gs[0, 0])
    psf_z = zoom_crop(psf_avg, com_avg, zoom)
    ax1.imshow(psf_z, cmap='inferno', vmin=0, vmax=psf_avg.max(),
               origin='lower', interpolation='nearest',
               extent=[-zoom, zoom, -zoom, zoom])
    ax1.set_title("Mean PSF (linear)")
    ax1.set_xlabel("px"); ax1.set_ylabel("px")

    # --- (0,1) Mean PSF zoomed, log ---
    ax2 = fig.add_subplot(gs[0, 1])
    log_z = zoom_crop(np.log10(psf_avg + 1), com_avg, zoom)
    ax2.imshow(log_z, cmap='inferno', origin='lower', interpolation='nearest',
               extent=[-zoom, zoom, -zoom, zoom])
    ax2.set_title("Mean PSF (log10)")
    ax2.set_xlabel("px"); ax2.set_ylabel("px")

    # --- (0,2) Std-dev map (variation) zoomed ---
    ax3 = fig.add_subplot(gs[0, 2])
    if n_images > 1:
        std_z = zoom_crop(psf_std, com_avg, zoom)
        im3 = ax3.imshow(std_z, cmap='magma', origin='lower', interpolation='nearest',
                         extent=[-zoom, zoom, -zoom, zoom])
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_title("PSF std-dev across images")
    else:
        if fit_img is not None:
            res_z = zoom_crop(psf_avg - fit_img, com_avg, zoom)
            lim = max(abs(res_z.min()), abs(res_z.max()))
            ax3.imshow(res_z, cmap='RdBu_r', vmin=-lim, vmax=lim,
                       origin='lower', interpolation='nearest',
                       extent=[-zoom, zoom, -zoom, zoom])
            ax3.set_title("Residual (PSF - Gauss)")
        else:
            ax3.text(0.5, 0.5, "Single image\nno variation", transform=ax3.transAxes,
                     ha='center', va='center')
            ax3.set_title("Variation")
    ax3.set_xlabel("px"); ax3.set_ylabel("px")

    # --- (1,0) Radial PSF profile with spread ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(radii_psf, radial_mean, 'k-', lw=1.5, label='Mean')
    if n_images > 1:
        ax4.fill_between(radii_psf,
                         np.clip(radial_mean - radial_std, 0, None),
                         radial_mean + radial_std,
                         alpha=0.25, color='steelblue', label='+/-1 std')
        for i in range(min(n_images, 20)):
            ax4.plot(radii_psf, all_radial_psf[i], color='gray', lw=0.3, alpha=0.4)
    if popt is not None:
        sigma_avg = (sx + sy) / 2
        gauss_r = np.exp(-radii_psf**2 / (2 * sigma_avg**2))
        ax4.plot(radii_psf, gauss_r, 'r--', lw=1, alpha=0.7,
                 label='Gauss (s=%.1f px)' % sigma_avg)
    ax4.set_xlabel("Radius (px)")
    ax4.set_ylabel("Normalised intensity")
    ax4.set_title("Radial PSF")
    ax4.legend(fontsize=7)
    ax4.set_xlim(0, min(args.crop_radius, 25))
    ax4.grid(True, alpha=0.3)

    # --- (1,1) Encircled energy with spread ---
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(radii_psf, ee_mean, 'b-', lw=1.5, label='Mean')
    if n_images > 1:
        ax5.fill_between(radii_psf,
                         np.clip(ee_mean - ee_std, 0, None),
                         np.clip(ee_mean + ee_std, None, 1.0),
                         alpha=0.25, color='steelblue', label='+/-1 std')
    for frac, ls in [(0.8, '--'), (0.9, ':')]:
        idx = np.searchsorted(ee_mean, frac)
        if idx < len(radii_psf):
            ax5.axhline(frac, color='gray', ls=ls, alpha=0.5)
            ax5.axvline(radii_psf[idx], color='gray', ls=ls, alpha=0.5)
            ax5.annotate("EE%d @ r=%d px" % (int(frac * 100), radii_psf[idx]),
                         (radii_psf[idx], frac), fontsize=7,
                         xytext=(5, -10), textcoords='offset points')
    ax5.set_xlabel("Radius (px)")
    ax5.set_ylabel("Encircled energy")
    ax5.set_title("Encircled Energy")
    ax5.set_xlim(0, min(args.crop_radius, 25))
    ax5.set_ylim(0, 1.05)
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)

    # --- (1,2) PSF cross-sections (on average) ---
    ax6 = fig.add_subplot(gs[1, 2])
    cy_int, cx_int = int(round(com_avg[0])), int(round(com_avg[1]))
    h_slice = psf_avg[cy_int, :]
    v_slice = psf_avg[:, cx_int]
    hmax = max(h_slice.max(), v_slice.max(), 1)
    ax6.plot(np.arange(len(h_slice)) - cx_int, h_slice / hmax, 'b-', lw=1, label='Horizontal')
    ax6.plot(np.arange(len(v_slice)) - cy_int, v_slice / hmax, 'r-', lw=1, label='Vertical')
    ax6.set_xlabel("Offset (px)")
    ax6.set_ylabel("Normalised intensity")
    ax6.set_title("PSF cross-sections (mean)")
    ax6.legend(fontsize=7)
    ax6.set_xlim(-zoom, zoom)
    ax6.grid(True, alpha=0.3)

    # --- (2,0) 2D MTF of mean PSF ---
    ax7 = fig.add_subplot(gs[2, 0])
    extent_half = 0.5
    s2 = mtf_2d_avg.shape[0]
    c2 = s2 // 2
    r_show = int(s2 * extent_half)
    mtf_crop = mtf_2d_avg[c2 - r_show:c2 + r_show, c2 - r_show:c2 + r_show]
    ax7.imshow(mtf_crop, cmap='viridis', vmin=0, vmax=1, origin='lower',
               extent=[-extent_half, extent_half, -extent_half, extent_half])
    ax7.set_xlabel("fx (cy/px)")
    ax7.set_ylabel("fy (cy/px)")
    ax7.set_title("2D MTF (mean PSF)")

    # --- (2,1) Radial MTF with spread ---
    ax8 = fig.add_subplot(gs[2, 1])
    valid = freq_ref <= nyquist
    ax8.plot(freq_ref[valid], mtf_mean[valid], 'k-', lw=1.5, label='Mean')
    if n_images > 1:
        ax8.fill_between(freq_ref[valid],
                         np.clip(mtf_mean[valid] - mtf_std[valid], 0, None),
                         np.clip(mtf_mean[valid] + mtf_std[valid], None, 1.05),
                         alpha=0.25, color='steelblue', label='+/-1 std')
        for i in range(min(n_images, 20)):
            ax8.plot(freq_ref[valid], all_mtf[i][valid], color='gray', lw=0.3, alpha=0.4)
    ax8.axhline(0.5, color='orange', ls='--', alpha=0.6,
                label='MTF50 = %.1f %s' % (mtf50, freq_label))
    ax8.axhline(0.1, color='red', ls='--', alpha=0.6,
                label='MTF10 = %.1f %s' % (mtf10, freq_label))
    if not np.isnan(mtf50):
        ax8.axvline(mtf50, color='orange', ls=':', alpha=0.4)
    if not np.isnan(mtf10):
        ax8.axvline(mtf10, color='red', ls=':', alpha=0.4)
    ax8.set_xlabel("Spatial frequency (%s)" % freq_label)
    ax8.set_ylabel("MTF")
    ax8.set_title("Radial MTF")
    ax8.set_xlim(0, nyquist)
    ax8.set_ylim(0, 1.05)
    ax8.legend(fontsize=7)
    ax8.grid(True, alpha=0.3)

    # --- (2,2) MTF H/V cross-sections with spread ---
    ax9 = fig.add_subplot(gs[2, 2])
    all_mtf_h = []
    all_mtf_v = []
    for psf in aligned_psfs:
        _, _, m2d, _, _ = compute_mtf(psf, args.pixel_pitch_um)
        sm = m2d.shape[0]
        cm = sm // 2
        all_mtf_h.append(m2d[cm, cm:])
        all_mtf_v.append(m2d[cm:, cm])

    min_len = min(len(h) for h in all_mtf_h)
    all_mtf_h = np.array([h[:min_len] for h in all_mtf_h])
    all_mtf_v = np.array([v[:min_len] for v in all_mtf_v])
    freq_1d = np.arange(min_len) / (min_len * 2)
    if args.pixel_pitch_um:
        freq_1d_plot = freq_1d / (args.pixel_pitch_um * 1e-3)
    else:
        freq_1d_plot = freq_1d
    valid_h = freq_1d_plot <= nyquist

    mtf_h_mean = all_mtf_h.mean(axis=0)
    mtf_v_mean = all_mtf_v.mean(axis=0)
    ax9.plot(freq_1d_plot[valid_h], mtf_h_mean[valid_h], 'b-', lw=1.2, label='H mean')
    ax9.plot(freq_1d_plot[valid_h], mtf_v_mean[valid_h], 'r-', lw=1.2, label='V mean')
    if n_images > 1:
        mtf_h_std = all_mtf_h.std(axis=0)
        mtf_v_std = all_mtf_v.std(axis=0)
        ax9.fill_between(freq_1d_plot[valid_h],
                         np.clip(mtf_h_mean[valid_h] - mtf_h_std[valid_h], 0, None),
                         np.clip(mtf_h_mean[valid_h] + mtf_h_std[valid_h], None, 1.05),
                         alpha=0.15, color='blue')
        ax9.fill_between(freq_1d_plot[valid_h],
                         np.clip(mtf_v_mean[valid_h] - mtf_v_std[valid_h], 0, None),
                         np.clip(mtf_v_mean[valid_h] + mtf_v_std[valid_h], None, 1.05),
                         alpha=0.15, color='red')
    ax9.set_xlabel("Spatial frequency (%s)" % freq_label)
    ax9.set_ylabel("MTF")
    ax9.set_title("MTF H/V")
    ax9.legend(fontsize=7)
    ax9.set_xlim(0, nyquist)
    ax9.set_ylim(0, 1.05)
    ax9.grid(True, alpha=0.3)

    plt.savefig(args.output, dpi=200)
    print("\nSaved figure to %s" % args.output)

    # ---- Save numerical data ----
    outdata = dict(
        psf_avg=psf_avg, psf_std=psf_std,
        radii_psf=radii_psf,
        radial_psf_mean=radial_mean, radial_psf_std=radial_std,
        freq=freq_ref, mtf_mean=mtf_mean, mtf_std=mtf_std,
        ee_mean=ee_mean, ee_std=ee_std,
        freq_label=np.array(freq_label),
        mtf50=mtf50, mtf10=mtf10, nyquist=nyquist,
        n_images=n_images,
        per_image_mtf50=all_mtf50, per_image_mtf10=all_mtf10,
    )
    if len(all_sx) > 0:
        outdata['per_image_sigma_x'] = np.array(all_sx)
        outdata['per_image_sigma_y'] = np.array(all_sy)

    npz_path = args.output.replace('.png', '_data.npz')
    np.savez(npz_path, **outdata)
    print("Saved numerical data to %s" % npz_path)


if __name__ == "__main__":
    main()