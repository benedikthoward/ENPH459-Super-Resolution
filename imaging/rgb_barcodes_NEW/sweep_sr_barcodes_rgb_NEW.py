#!/usr/bin/env python3
"""
sweep_sr_barcodes_rgb_NEW.py

Super-resolution processing for all sessions in imaging/rgb_barcodes_NEW/.

Data layout
-----------
  imaging/rgb_barcodes_NEW/<session>/
      color_nominal_1.0px_settle50ms/                  ← nominal calibration
      color_special_<timestamp>_1.0px_settle50ms/      ← special calibrations (×4)
          corner0_rep00.png   ← (-x,+y)
          corner1_rep00.png   ← (+x,+y)
          corner2_rep00.png   ← (-x,-y)
          corner3_rep00.png   ← (+x,-y)
          corner0_rep01.png   ← rep 1
          ...
          metadata.json

Images are 1536×2048 uint8, stored as raw Bayer grayscale (RGGB pattern).

Red channel extraction
----------------------
Red pixels: img[0::2, 0::2]  →  768 × 1024 LR pixels
1 sensor pixel shift  ≡  0.5 red-channel LR pixel
Upsample factor 2  →  1536 × 2048 HR (= full sensor resolution)

Shift geometry
--------------
Nominal 1.0 sensor-pixel diagonal shifts → 0.5 red-LR-pixel shifts:
  corner0 (-x,+y)  →  (dy, dx) = (+0.5, -0.5) LR px
  corner1 (+x,+y)  →  (dy, dx) = (+0.5, +0.5) LR px
  corner2 (-x,-y)  →  (dy, dx) = (-0.5, -0.5) LR px
  corner3 (+x,-y)  →  (dy, dx) = (-0.5, +0.5) LR px

Each calibration set × repetition is processed independently.

SR methods
----------
  Native-2x  : bicubic 2× upsample of the mean of all 4 red-channel LR frames
  SAA        : Shift-and-Add using nominal 0.5 LR-px shifts
  SAA+IBP    : IBP (iterative back-projection) initialised from SAA

Output
------
  imaging/rgb_barcodes_NEW/<session>/sr_output/<calib_dir_name>/rep00/
      native_2x.png
      SAA.png
      SAA_IBP.png
      LR_red_mean.png
      comparison.png
      convergence.png
"""

import os
import json
import time
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import shift as ndi_shift, zoom as ndi_zoom
from scipy.signal import fftconvolve

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT     = SCRIPT_DIR  # script lives inside rgb_barcodes_NEW/
PSF_CALIB_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'calibration', 'psf_images')

# ── Sensor / Bayer ─────────────────────────────────────────────────────────────

BAYER_ROW_OFFSET = 0   # red pixels at even rows
BAYER_COL_OFFSET = 0   # red pixels at even cols
UPSAMPLE_FACTOR  = 2
f                = UPSAMPLE_FACTOR

# ── IBP parameters ─────────────────────────────────────────────────────────────

PSF_HALFWIDTH  = 3
IBP_ITERATIONS = 80
IBP_STEP_SIZE  = 0.5

# 4 diagonal corners and their nominal (dy, dx) shifts in red-channel LR pixels
# (sensor-pixel shift = 1.0 px  →  red-LR shift = 0.5 px)
CORNER_POSITIONS = [
    (0, '(-x,+y)', (+0.5, -0.5)),
    (1, '(+x,+y)', (+0.5, +0.5)),
    (2, '(-x,-y)', (-0.5, -0.5)),
    (3, '(+x,-y)', (-0.5, +0.5)),
]

COLORS = {'Native-2x': 'C0', 'SAA': 'C2', 'SAA+IBP': 'C3'}


# ── Image helpers ──────────────────────────────────────────────────────────────

def load_gray(path):
    img = np.array(Image.open(path), dtype=np.float64)
    return img.mean(axis=2) if img.ndim == 3 else img


def extract_red(img):
    """Extract the red Bayer channel (even rows, even cols) from RGGB image."""
    return img[BAYER_ROW_OFFSET::2, BAYER_COL_OFFSET::2].copy()


def find_calib_dirs(session_dir):
    """Return sorted list of color_* calibration sub-directories."""
    dirs = sorted([
        os.path.join(session_dir, d) for d in os.listdir(session_dir)
        if d.startswith('color_') and os.path.isdir(os.path.join(session_dir, d))
    ])
    if not dirs:
        raise FileNotFoundError(f'No color_* sub-dirs in {session_dir}')
    return dirs


def find_rep_indices(calib_dir):
    """Return sorted list of rep indices found in the calibration dir."""
    reps = set()
    for fname in os.listdir(calib_dir):
        if fname.startswith('corner') and fname.endswith('.png'):
            rep_str = fname.split('_rep')[1].split('.')[0]
            reps.add(int(rep_str))
    return sorted(reps)


def load_run(calib_dir, rep_idx):
    """
    Load the 4 corner red-channel LR frames for one calibration × one rep.

    Verifies corner labels against metadata.json if present.

    Returns:
        lr_frames : list of 4 LR arrays
        shifts_lr : list of 4 (dy_lr, dx_lr) tuples in red-channel LR pixels
    """
    meta_path = os.path.join(calib_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        for pos in meta['positions']:
            expected_label = CORNER_POSITIONS[pos['index']][1]
            assert pos['label'] == expected_label, \
                f"Corner {pos['index']} label mismatch: {pos['label']} != {expected_label}"

    lr_frames = []
    shifts_lr = []
    for corner_idx, label, (dy, dx) in CORNER_POSITIONS:
        fname = f'corner{corner_idx}_rep{rep_idx:02d}.png'
        path = os.path.join(calib_dir, fname)
        lr_frames.append(extract_red(load_gray(path)))
        shifts_lr.append((dy, dx))
        print(f'    corner{corner_idx} {label:10s}: '
              f'shift_lr = (dy={dy:+.1f}, dx={dx:+.1f}) red-LR px')

    return lr_frames, shifts_lr


# ── PSF loading ────────────────────────────────────────────────────────────────

def load_psf():
    """
    Find all pos4_(0,0).png files under PSF_CALIB_DIR, align each by its
    peak location, average the aligned patches, then extract the PSF kernel.
    """
    MARGIN = PSF_HALFWIDTH + 6

    patches = []
    for sweep_dir in sorted(os.listdir(PSF_CALIB_DIR)):
        full = os.path.join(PSF_CALIB_DIR, sweep_dir)
        if not os.path.isdir(full):
            continue
        pos4_path = os.path.join(full, 'pos4_(0,0).png')
        if not os.path.exists(pos4_path):
            continue
        img = load_gray(pos4_path)
        pr, pc = np.unravel_index(img.argmax(), img.shape)
        R = MARGIN
        if pr < R or pr + R + 1 > img.shape[0] or pc < R or pc + R + 1 > img.shape[1]:
            print(f'  PSF skip (peak too close to edge): {pos4_path}')
            continue
        patches.append(img[pr - R:pr + R + 1, pc - R:pc + R + 1].copy())

    if not patches:
        raise FileNotFoundError(f'No pos4_(0,0).png found under {PSF_CALIB_DIR}')

    avg = np.mean(patches, axis=0)
    print(f'  PSF: averaged {len(patches)} pos4_(0,0).png images (peak-aligned)')

    R = MARGIN
    kernel = avg[R - PSF_HALFWIDTH:R + PSF_HALFWIDTH + 1,
                 R - PSF_HALFWIDTH:R + PSF_HALFWIDTH + 1].copy()

    corners = np.concatenate([
        kernel[:3, :3].ravel(), kernel[:3, -3:].ravel(),
        kernel[-3:, :3].ravel(), kernel[-3:, -3:].ravel(),
    ])
    kernel -= np.mean(corners)
    kernel  = np.clip(kernel, 0, None)
    kernel /= kernel.sum()
    print(f'  PSF kernel shape: {kernel.shape}')
    return kernel


def make_gaussian_psf(halfwidth, sigma):
    """Create a normalized 2D Gaussian PSF kernel."""
    size = 2 * halfwidth + 1
    y, x = np.mgrid[-halfwidth:halfwidth+1, -halfwidth:halfwidth+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    print(f'  Gaussian PSF: {size}×{size}, sigma={sigma:.2f}')
    return kernel


# ── SR core ────────────────────────────────────────────────────────────────────

def blur(img, kernel):
    return fftconvolve(img, kernel, mode='same')


def forward_model(hr, kernel, shift_yx, factor):
    blurred = blur(hr, kernel)
    shifted = ndi_shift(blurred, (shift_yx[0] * factor, shift_yx[1] * factor),
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
    shifted = ndi_shift(up, (-shift_yx[0] * factor, -shift_yx[1] * factor),
                        order=3, mode='nearest')
    return blur(shifted, kernel[::-1, ::-1])


def shift_and_add(lr_list, shifts_yx, factor=2, order=3):
    h_lr, w_lr = lr_list[0].shape
    acc = np.zeros((h_lr * factor, w_lr * factor))
    for lr, (dy, dx) in zip(lr_list, shifts_yx):
        up  = ndi_zoom(lr, factor, order=order)
        acc += ndi_shift(up, (dy * factor, dx * factor), order=3, mode='nearest')
    return acc / len(lr_list)


def ibp(lr_list, shifts_yx, kernel, hr_init, factor=2, n_iter=80, step=0.5):
    hr = hr_init.copy()
    n  = len(lr_list)
    errors = []
    for it in range(n_iter):
        correction = np.zeros_like(hr)
        total_err  = 0.0
        for lr, s in zip(lr_list, shifts_yx):
            sim = forward_model(hr, kernel, s, factor)
            mh  = min(sim.shape[0], lr.shape[0])
            mw  = min(sim.shape[1], lr.shape[1])
            err = lr[:mh, :mw] - sim[:mh, :mw]
            total_err += np.mean(err ** 2)
            correction += back_project(err, kernel, s, factor, hr.shape)
        hr += step * correction / n
        hr  = np.clip(hr, 0, 255)
        errors.append(total_err / n)
        if (it + 1) % 10 == 0:
            print(f'    iter {it+1:3d}/{n_iter}  MSE = {errors[-1]:.4f}')
    return hr, errors


# ── Plotting ───────────────────────────────────────────────────────────────────

def save_comparison(hr_images, mean_lr, out_dir, title):
    H, W = list(hr_images.values())[0].shape
    cr = slice(H // 2 - 100, H // 2 + 100)
    cc = slice(W // 2 - 100, W // 2 + 100)

    n   = len(hr_images) + 1
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    ds = 4
    axes[0, 0].imshow(mean_lr[::ds, ::ds], cmap='gray', interpolation='nearest')
    axes[0, 0].set_title('LR red (mean)', fontsize=9)
    axes[0, 0].axis('off')
    for i, (name, img) in enumerate(hr_images.items(), 1):
        axes[0, i].imshow(img[::ds * 2, ::ds * 2], cmap='gray', interpolation='nearest')
        axes[0, i].set_title(name, fontsize=9)
        axes[0, i].axis('off')

    lr_zoom = ndi_zoom(mean_lr, 2, order=3)
    axes[1, 0].imshow(lr_zoom[cr, cc], cmap='gray', interpolation='nearest')
    axes[1, 0].set_title('LR bicubic 2× (mean)', fontsize=8)
    axes[1, 0].axis('off')
    for i, (name, img) in enumerate(hr_images.items(), 1):
        axes[1, i].imshow(img[cr, cc], cmap='gray', interpolation='nearest')
        axes[1, i].set_title(name, fontsize=8)
        axes[1, i].axis('off')

    plt.suptitle(f'Red channel SR — {title}', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison.png'), bbox_inches='tight', dpi=100)
    plt.close(fig)
    print('  Saved comparison.png')


def save_convergence(ibp_errors, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ibp_errors, color=COLORS['SAA+IBP'], lw=1.5)
    ax.set_title('IBP convergence')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'convergence.png'), bbox_inches='tight')
    plt.close(fig)
    print('  Saved convergence.png')


# ── Per-run processing ────────────────────────────────────────────────────────

def process_run(calib_dir, rep_idx, session_dir, psf_kernel, output_root='sr_output'):
    calib_name = os.path.basename(calib_dir)
    out_dir    = os.path.join(session_dir, output_root, calib_name, f'rep{rep_idx:02d}')
    done_flag  = os.path.join(out_dir, 'done.flag')
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(done_flag):
        print(f'  [skip] {calib_name}/rep{rep_idx:02d} — already done')
        return

    print(f'\n  Loading {calib_name} / rep{rep_idx:02d}')
    lr_frames, shifts_lr = load_run(calib_dir, rep_idx)
    mean_lr = np.mean(lr_frames, axis=0)
    print(f'  LR red frame shape: {mean_lr.shape}')

    print('  Computing Native-2x ...')
    t0 = time.time()
    native_hr = ndi_zoom(mean_lr, f, order=3)
    print(f'  Done {time.time()-t0:.1f}s  shape={native_hr.shape}')

    print('  Computing SAA ...')
    t0 = time.time()
    saa_hr = shift_and_add(lr_frames, shifts_lr, factor=f, order=3)
    print(f'  Done {time.time()-t0:.1f}s')

    print(f'  Computing SAA+IBP ({IBP_ITERATIONS} iters) ...')
    t0 = time.time()
    ibp_hr, ibp_errors = ibp(
        lr_frames, shifts_lr, psf_kernel, saa_hr.copy(),
        factor=f, n_iter=IBP_ITERATIONS, step=IBP_STEP_SIZE,
    )
    print(f'  Done {time.time()-t0:.1f}s')

    hr_images = {
        'Native-2x': native_hr,
        'SAA':       saa_hr,
        'SAA+IBP':   ibp_hr,
    }

    name_map = {'Native-2x': 'native_2x', 'SAA': 'SAA', 'SAA+IBP': 'SAA_IBP'}
    for name, img in hr_images.items():
        arr  = np.clip(img, 0, 255).astype(np.uint8)
        path = os.path.join(out_dir, f'{name_map[name]}.png')
        Image.fromarray(arr).save(path)
        print(f'  Saved {name_map[name]}.png')

    arr_lr = np.clip(mean_lr, 0, 255).astype(np.uint8)
    Image.fromarray(arr_lr).save(os.path.join(out_dir, 'LR_red_mean.png'))
    print('  Saved LR_red_mean.png')

    title = f'{os.path.basename(session_dir)} / {calib_name} / rep{rep_idx:02d}'
    save_comparison(hr_images, mean_lr, out_dir, title)
    save_convergence(ibp_errors, out_dir)
    open(done_flag, 'w').close()
    print(f'  Output: {out_dir}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SR processing for rgb_barcodes_NEW')
    parser.add_argument('--gaussian', action='store_true',
                        help='Use a Gaussian PSF instead of the measured PSF')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Sigma for Gaussian PSF (default: 1.0)')
    parser.add_argument('--psf-size', type=int, default=7,
                        help='Gaussian PSF kernel size in pixels (default: 7)')
    args = parser.parse_args()

    # Only treat directories that contain color_* sub-dirs as sessions
    sessions = sorted([
        os.path.join(DATA_ROOT, d)
        for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
        and any(c.startswith('color_') for c in os.listdir(os.path.join(DATA_ROOT, d)))
    ])
    if not sessions:
        raise FileNotFoundError(f'No session folders found in {DATA_ROOT}')

    print(f'Found {len(sessions)} session(s):\n' +
          '\n'.join(f'  {os.path.basename(s)}' for s in sessions))

    if args.gaussian:
        halfwidth = (args.psf_size - 1) // 2
        print(f'\nUsing Gaussian PSF (size={args.psf_size}, sigma={args.sigma}) ...')
        psf_kernel = make_gaussian_psf(halfwidth, args.sigma)
        output_root = f'sr_output_gaussian_s{args.sigma}'
    else:
        print('\nLoading measured PSF kernel ...')
        psf_kernel = load_psf()
        output_root = 'sr_output'

    t_total = time.time()
    run_count = 0
    for session_dir in sessions:
        session_name = os.path.basename(session_dir)
        calib_dirs = find_calib_dirs(session_dir)
        print(f'\n{"="*60}')
        print(f'Session: {session_name}')
        print(f'  {len(calib_dirs)} calibration set(s)')

        for calib_dir in calib_dirs:
            rep_indices = find_rep_indices(calib_dir)
            for rep_idx in rep_indices:
                run_count += 1
                print(f'\n[Run {run_count}] ', end='')
                process_run(calib_dir, rep_idx, session_dir, psf_kernel, output_root)

    print(f'\nAll {run_count} runs done in {(time.time()-t_total)/60:.1f} min')


if __name__ == '__main__':
    main()
