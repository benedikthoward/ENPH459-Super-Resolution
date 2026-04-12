#!/usr/bin/env python3
"""
sweep_sr_cal_target.py

Super-resolution processing for the mono calibration target images.

Data layout
-----------
  imaging/mono_cal_target/<session>/
      center.png              <- unshifted reference
      shift_0.png             <- corner shift (+0.5, -0.5) px
      shift_1.png             <- corner shift (+0.5, +0.5) px
      shift_2.png             <- corner shift (-0.5, -0.5) px
      shift_3.png             <- corner shift (-0.5, +0.5) px

Images are 1536x2048 uint8 mono.

Shift geometry
--------------
5 images: 1 centre + 4 diagonal corners at nominal 0.5 sensor-pixel shifts:
  center   ->  (dy, dx) = ( 0.0,  0.0) px
  shift_0  ->  (dy, dx) = (+0.5, -0.5) px
  shift_1  ->  (dy, dx) = (+0.5, +0.5) px
  shift_2  ->  (dy, dx) = (-0.5, -0.5) px
  shift_3  ->  (dy, dx) = (-0.5, +0.5) px

Upsample factor 2 -> 3072 x 4096 HR output.

PSF
---
Loaded from imaging/calibration/psf_images/ (averaged pos4_(0,0).png).

SR methods
----------
  Native-2x  : bicubic 2x upsample of the mean of all 5 LR frames
  SAA        : Shift-and-Add using nominal shifts
  SAA+IBP    : IBP (iterative back-projection) initialised from SAA

Output
------
  imaging/mono_cal_target/<session>/sr_output/
      native_2x.png
      SAA.png
      SAA_IBP.png
      LR_mean.png
      comparison.png
      convergence.png
"""

import os
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

# -- Paths -----------------------------------------------------------------

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))

# -- SR parameters ---------------------------------------------------------

UPSAMPLE_FACTOR = 2
f               = UPSAMPLE_FACTOR

PSF_SIZE        = 7
PSF_SIGMA       = 1.0
IBP_ITERATIONS  = 80
IBP_STEP_SIZE   = 0.5

# Image filenames and their nominal (dy, dx) shifts in sensor/LR pixels
IMAGE_SHIFTS = [
    ('center.png',  ( 0.0,  0.0)),
    ('shift_0.png', (+0.5, -0.5)),
    ('shift_1.png', (+0.5, +0.5)),
    ('shift_2.png', (-0.5, -0.5)),
    ('shift_3.png', (-0.5, +0.5)),
]

COLORS = {'Native-2x': 'C0', 'SAA': 'C2', 'SAA+IBP': 'C3'}


# -- Image helpers ---------------------------------------------------------

def load_gray(path):
    img = np.array(Image.open(path), dtype=np.float64)
    return img.mean(axis=2) if img.ndim == 3 else img


def load_session(session_dir):
    """
    Load all LR frames for a session.

    Returns:
        lr_frames  : list of LR arrays
        shifts_lr  : list of (dy_lr, dx_lr) tuples in sensor pixel units
    """
    lr_frames = []
    shifts_lr = []
    for fname, (dy, dx) in IMAGE_SHIFTS:
        path = os.path.join(session_dir, fname)
        if not os.path.exists(path):
            print(f'  WARNING: {fname} not found, skipping')
            continue
        lr_frames.append(load_gray(path))
        shifts_lr.append((dy, dx))
        print(f'    {fname:16s}: shift_lr = (dy={dy:+.1f}, dx={dx:+.1f}) px')

    if len(lr_frames) < 2:
        raise FileNotFoundError(f'Need at least 2 images in {session_dir}')

    return lr_frames, shifts_lr


# -- PSF loading -----------------------------------------------------------

def make_gaussian_psf(size=PSF_SIZE, sigma=PSF_SIGMA):
    """Create a normalized 2D Gaussian PSF kernel."""
    hw = size // 2
    y, x = np.mgrid[-hw:hw + 1, -hw:hw + 1].astype(np.float64)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    print(f'  PSF: Gaussian {size}x{size}, sigma={sigma}')
    return kernel


# -- SR core ---------------------------------------------------------------

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


# -- Plotting --------------------------------------------------------------

def save_comparison(hr_images, mean_lr, out_dir, title):
    H, W = list(hr_images.values())[0].shape
    cr = slice(H // 2 - 100, H // 2 + 100)
    cc = slice(W // 2 - 100, W // 2 + 100)

    n   = len(hr_images) + 1
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    ds = 4
    axes[0, 0].imshow(mean_lr[::ds, ::ds], cmap='gray', interpolation='nearest')
    axes[0, 0].set_title('LR mean', fontsize=9)
    axes[0, 0].axis('off')
    for i, (name, img) in enumerate(hr_images.items(), 1):
        axes[0, i].imshow(img[::ds * 2, ::ds * 2], cmap='gray', interpolation='nearest')
        axes[0, i].set_title(name, fontsize=9)
        axes[0, i].axis('off')

    lr_zoom = ndi_zoom(mean_lr, 2, order=3)
    axes[1, 0].imshow(lr_zoom[cr, cc], cmap='gray', interpolation='nearest')
    axes[1, 0].set_title('LR bicubic 2x (mean)', fontsize=8)
    axes[1, 0].axis('off')
    for i, (name, img) in enumerate(hr_images.items(), 1):
        axes[1, i].imshow(img[cr, cc], cmap='gray', interpolation='nearest')
        axes[1, i].set_title(name, fontsize=8)
        axes[1, i].axis('off')

    plt.suptitle(f'Mono SR Cal Target - {title}', fontsize=10)
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


# -- Per-session processing ------------------------------------------------

def process_session(session_dir, psf_kernel):
    session_name = os.path.basename(session_dir)
    out_dir      = os.path.join(session_dir, 'sr_output')
    done_flag    = os.path.join(out_dir, 'done.flag')
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(done_flag):
        print(f'  [skip] {session_name} - already done')
        return

    print(f'\nLoading session: {session_name}')
    lr_frames, shifts_lr = load_session(session_dir)
    mean_lr = np.mean(lr_frames, axis=0)
    print(f'  LR frame shape: {mean_lr.shape}  ({len(lr_frames)} images)')

    # Native-2x
    print('  Computing Native-2x ...')
    t0 = time.time()
    native_hr = ndi_zoom(mean_lr, f, order=3)
    print(f'  Done {time.time()-t0:.1f}s  shape={native_hr.shape}')

    # SAA
    print('  Computing SAA ...')
    t0 = time.time()
    saa_hr = shift_and_add(lr_frames, shifts_lr, factor=f, order=3)
    print(f'  Done {time.time()-t0:.1f}s')

    # SAA + IBP
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

    # Save PNGs
    name_map = {'Native-2x': 'native_2x', 'SAA': 'SAA', 'SAA+IBP': 'SAA_IBP'}
    for name, img in hr_images.items():
        arr  = np.clip(img, 0, 255).astype(np.uint8)
        path = os.path.join(out_dir, f'{name_map[name]}.png')
        Image.fromarray(arr).save(path)
        print(f'  Saved {name_map[name]}.png')

    arr_lr = np.clip(mean_lr, 0, 255).astype(np.uint8)
    Image.fromarray(arr_lr).save(os.path.join(out_dir, 'LR_mean.png'))
    print('  Saved LR_mean.png')

    save_comparison(hr_images, mean_lr, out_dir, session_name)
    save_convergence(ibp_errors, out_dir)
    open(done_flag, 'w').close()
    print(f'  Output: {out_dir}')


# -- Main ------------------------------------------------------------------

def main():
    sessions = sorted([
        os.path.join(SCRIPT_DIR, d)
        for d in os.listdir(SCRIPT_DIR)
        if os.path.isdir(os.path.join(SCRIPT_DIR, d))
            and os.path.exists(os.path.join(SCRIPT_DIR, d, 'center.png'))
    ])
    if not sessions:
        raise FileNotFoundError(f'No session folders with center.png found in {SCRIPT_DIR}')

    print(f'Found {len(sessions)} session(s):\n' +
          '\n'.join(f'  {os.path.basename(s)}' for s in sessions))

    print('\nCreating Gaussian PSF kernel ...')
    psf_kernel = make_gaussian_psf()

    t_total = time.time()
    for i, session_dir in enumerate(sessions, 1):
        print(f'\n[{i}/{len(sessions)}] ', end='')
        process_session(session_dir, psf_kernel)

    print(f'\nAll sessions done in {(time.time()-t_total)/60:.1f} min')


if __name__ == '__main__':
    main()
