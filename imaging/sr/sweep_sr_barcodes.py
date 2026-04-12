#!/usr/bin/env python3
"""
sweep_sr_barcodes.py

Super-resolution processing for the rgb_barcodes_2-6mil dataset.

Data layout
-----------
  imaging/rgb_barcodes_2-6mil/special_1.0px_settle50ms/
      corner{0-3}_rep{00-04}.png   (5 repeats × 4 shift positions)
      metadata.json                (expected shifts in sensor pixels)

The images are raw Bayer grayscale (RGGB pattern).
Red channel: img[0::2, 0::2]  →  768 × 1024 LR pixels
Nominal sensor-pixel shift = 1.0 px  →  0.5 red-channel LR pixel
Upsample factor 2  →  1536 × 2048 HR

PSF
---
Loaded from imaging/calibration/psf_images/sweepx_tilt0.30000deg/pos4_(0,0).png
(single unshifted pinhole image at a mid-range tilt angle).

SR methods
----------
  Native-2x  : bicubic 2× upsample of the mean of all 4 LR frames
  SAA        : Shift-and-Add using expected shifts from metadata
  SAA+IBP    : IBP (iterative back-projection) initialised from SAA

Output
------
  imaging/rgb_barcodes_2-6mil/sr_output/
      native_2x.png
      SAA.png
      SAA_IBP.png
      LR_red_mean.png
      shifts.json
      comparison.png
      convergence.png
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
from scipy.ndimage import shift as ndi_shift, zoom as ndi_zoom
from scipy.signal import fftconvolve

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT    = os.path.join(os.path.dirname(SCRIPT_DIR), 'rgb_barcodes_2-6mil')
INPUT_DIR    = os.path.join(DATA_ROOT, 'special_1.0px_settle50ms')
OUTPUT_DIR   = os.path.join(DATA_ROOT, 'sr_output')
# Single unshifted pinhole image used as PSF — mid-range tilt, pos4 = (0,0) position
PSF_IMAGE_PATH = os.path.join(
    os.path.dirname(SCRIPT_DIR), 'calibration', 'psf_images',
    'sweepx_tilt0.30000deg', 'pos4_(0,0).png'
)

# ── Sensor / Bayer ─────────────────────────────────────────────────────────────

BAYER_ROW_OFFSET = 0   # red pixels at even rows
BAYER_COL_OFFSET = 0   # red pixels at even cols
UPSAMPLE_FACTOR  = 2
f                = UPSAMPLE_FACTOR

# ── IBP parameters ─────────────────────────────────────────────────────────────

PSF_HALFWIDTH  = 3
IBP_ITERATIONS = 100
IBP_STEP_SIZE  = 0.5

# Corner label order — must match metadata.json corner pattern
CORNER_ORDER = ['(-x,+y)', '(+x,+y)', '(-x,-y)', '(+x,-y)']

COLORS = {'Native-2x': 'C0', 'SAA': 'C2', 'SAA+IBP': 'C3'}


# ── Image helpers ──────────────────────────────────────────────────────────────

def load_gray(path):
    img = np.array(Image.open(path), dtype=np.float64)
    return img.mean(axis=2) if img.ndim == 3 else img


def extract_red(img):
    """Extract the red Bayer channel (even rows, even cols) from RGGB image."""
    return img[BAYER_ROW_OFFSET::2, BAYER_COL_OFFSET::2].copy()


def load_combo(combo_dir):
    """
    Load 4-corner image set from a combo directory.

    Returns:
        lr_frames : list of 4 red-channel LR arrays (averaged over reps)
        shifts_lr : list of 4 (dy_lr, dx_lr) tuples in red-channel LR pixel units
        meta      : parsed metadata dict
    """
    with open(os.path.join(combo_dir, 'metadata.json')) as fp:
        meta = json.load(fp)

    def get_shift(label):
        if 'expected_shifts' in meta:
            s = meta['expected_shifts'][label]
            return s['dy_px'] / 2.0, s['dx_px'] / 2.0
        elif 'corners' in meta:
            c = meta['corners'][label]
            return c['expected_dy_px'] / 2.0, c['expected_dx_px'] / 2.0
        else:
            raise KeyError(f'Cannot find shift for {label} in metadata')

    lr_frames = []
    shifts_lr = []
    for idx, label in enumerate(CORNER_ORDER):
        reps = sorted([
            fn for fn in os.listdir(combo_dir)
            if fn.startswith(f'corner{idx}_rep') and fn.endswith('.png')
        ])
        if not reps:
            raise FileNotFoundError(f'No images for corner{idx} in {combo_dir}')
        stack = np.stack([
            extract_red(load_gray(os.path.join(combo_dir, r))) for r in reps
        ])
        lr_frames.append(stack.mean(axis=0))
        shifts_lr.append(get_shift(label))
        dy, dx = shifts_lr[-1]
        print(f'  corner{idx} ({label:10s}): {len(reps)} reps  '
              f'shift_lr = (dy={dy:+.4f}, dx={dx:+.4f}) px')

    return lr_frames, shifts_lr, meta


# ── PSF loading ────────────────────────────────────────────────────────────────

def load_psf():
    """
    Load PSF from a single pinhole image (pos4_(0,0).png at sweepx_tilt0.30000deg).
    Extract a (2*PSF_HALFWIDTH+1)² kernel centred on the peak.
    Falls back to a Gaussian if the file is not found.
    """
    if not os.path.exists(PSF_IMAGE_PATH):
        sigma = 1.0
        print(f'  PSF image not found at {PSF_IMAGE_PATH}')
        print(f'  Falling back to Gaussian (sigma={sigma} px)')
        size   = 2 * PSF_HALFWIDTH + 1
        ax     = np.arange(size) - PSF_HALFWIDTH
        g1d    = np.exp(-0.5 * (ax / sigma) ** 2)
        kernel = np.outer(g1d, g1d)
        kernel /= kernel.sum()
        return kernel

    print(f'  PSF: {PSF_IMAGE_PATH}')
    avg  = load_gray(PSF_IMAGE_PATH)
    peak = np.unravel_index(avg.argmax(), avg.shape)
    pr, pc = peak
    R      = PSF_HALFWIDTH
    kernel = avg[pr - R:pr + R + 1, pc - R:pc + R + 1].copy()

    # Background subtract and normalise
    corners = np.concatenate([
        kernel[:3, :3].ravel(), kernel[:3, -3:].ravel(),
        kernel[-3:, :3].ravel(), kernel[-3:, -3:].ravel(),
    ])
    kernel -= np.mean(corners)
    kernel  = np.clip(kernel, 0, None)
    kernel /= kernel.sum()

    print(f'  PSF kernel shape: {kernel.shape}')
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


def ibp(lr_list, shifts_yx, kernel, hr_init, factor=2, n_iter=20, step=0.5):
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
        if (it + 1) % 5 == 0:
            print(f'    iter {it+1:3d}/{n_iter}  MSE = {errors[-1]:.4f}')
    return hr, errors


# ── Plotting ───────────────────────────────────────────────────────────────────

def save_comparison(hr_images, mean_lr, out_dir):
    H, W = list(hr_images.values())[0].shape
    cr = slice(H // 2 - 100, H // 2 + 100)
    cc = slice(W // 2 - 100, W // 2 + 100)

    n   = len(hr_images) + 1
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    ds = 4
    axes[0, 0].imshow(mean_lr[::ds, ::ds], cmap='gray', interpolation='nearest')
    axes[0, 0].set_title('LR red (avg reps)', fontsize=9)
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

    plt.suptitle('Red channel SR — barcodes 2-6 mil', fontsize=11)
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading PSF kernel ...')
    psf_kernel = load_psf()

    print(f'\nLoading images from:\n  {INPUT_DIR}')
    lr_frames, shifts_lr, meta = load_combo(INPUT_DIR)
    mean_lr = np.mean(lr_frames, axis=0)
    print(f'  LR frame shape: {mean_lr.shape}')

    # ── Native-2x ─────────────────────────────────────────
    print('\nComputing Native-2x (bicubic upsample of mean LR) ...')
    t0 = time.time()
    native_hr = ndi_zoom(mean_lr, f, order=3)
    print(f'  Done in {time.time()-t0:.1f}s  shape={native_hr.shape}')

    # ── SAA ───────────────────────────────────────────────
    print('\nComputing SAA ...')
    t0 = time.time()
    saa_hr = shift_and_add(lr_frames, shifts_lr, factor=f, order=3)
    print(f'  Done in {time.time()-t0:.1f}s  shape={saa_hr.shape}')

    # ── SAA + IBP ─────────────────────────────────────────
    print(f'\nComputing SAA+IBP ({IBP_ITERATIONS} iterations) ...')
    t0 = time.time()
    ibp_hr, ibp_errors = ibp(
        lr_frames, shifts_lr, psf_kernel, saa_hr.copy(),
        factor=f, n_iter=IBP_ITERATIONS, step=IBP_STEP_SIZE,
    )
    print(f'  Done in {time.time()-t0:.1f}s')

    hr_images = {
        'Native-2x': native_hr,
        'SAA':       saa_hr,
        'SAA+IBP':   ibp_hr,
    }

    # ── Save outputs ──────────────────────────────────────
    print('\nSaving outputs ...')
    name_map = {'Native-2x': 'native_2x', 'SAA': 'SAA', 'SAA+IBP': 'SAA_IBP'}
    for name, img in hr_images.items():
        arr  = np.clip(img, 0, 255).astype(np.uint8)
        path = os.path.join(OUTPUT_DIR, f'{name_map[name]}.png')
        Image.fromarray(arr).save(path)
        print(f'  Saved {name_map[name]}.png')

    arr_lr = np.clip(mean_lr, 0, 255).astype(np.uint8)
    Image.fromarray(arr_lr).save(os.path.join(OUTPUT_DIR, 'LR_red_mean.png'))
    print('  Saved LR_red_mean.png')

    with open(os.path.join(OUTPUT_DIR, 'shifts.json'), 'w') as fp:
        json.dump({'shifts_lr_yx': shifts_lr, 'corner_labels': CORNER_ORDER}, fp, indent=2)
    print('  Saved shifts.json')

    save_comparison(hr_images, mean_lr, OUTPUT_DIR)
    save_convergence(ibp_errors, OUTPUT_DIR)

    print(f'\nAll outputs saved to:\n  {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
