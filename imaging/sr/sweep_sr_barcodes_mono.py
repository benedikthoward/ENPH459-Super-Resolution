#!/usr/bin/env python3
"""
sweep_sr_barcodes_mono.py

Super-resolution processing for all sessions in imaging/mono_barcodes/.

Data layout
-----------
  imaging/mono_barcodes/<session>/
      dist+0.0mm_lap*/                   ← best-focus distance folder
          shift*_0.50px_cycle0_rep0/     ← 9 images (3×3 grid), rep 0
          shift*_0.50px_cycle0_rep1/     ← same positions, rep 1
              pos0_(-x,+y).png
              pos2_(+x,+y).png
              pos6_(-x,-y).png
              pos8_(+x,-y).png
              ...

Images are 1536×2048 uint8 mono (no Bayer pattern).

Shift geometry
--------------
The 4 diagonal corners give 2-D coverage with nominal 0.5 sensor-pixel shifts:
  pos0 (-x,+y)  →  (dy, dx) = (+0.5, -0.5) px
  pos2 (+x,+y)  →  (dy, dx) = (+0.5, +0.5) px
  pos6 (-x,-y)  →  (dy, dx) = (-0.5, -0.5) px
  pos8 (+x,-y)  →  (dy, dx) = (-0.5, +0.5) px

For mono images 1 sensor pixel = 1 LR pixel, so shifts are used directly.
Upsample factor 2  →  3072 × 4096 HR output.

PSF
---
Loaded from imaging/calibration/psf_images/sweepx_tilt0.30000deg/pos4_(0,0).png
(single unshifted pinhole image).

SR methods
----------
  Native-2x  : bicubic 2× upsample of the mean of all 4 LR frames
  SAA        : Shift-and-Add using nominal 0.5 px shifts
  SAA+IBP    : IBP (iterative back-projection) initialised from SAA

Output
------
  imaging/mono_barcodes/<session>/sr_output/
      native_2x.png
      SAA.png
      SAA_IBP.png
      LR_mean.png
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
DATA_ROOT    = os.path.join(os.path.dirname(SCRIPT_DIR), 'mono_barcodes')
PSF_CALIB_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'calibration', 'psf_images')

# ── SR parameters ──────────────────────────────────────────────────────────────

UPSAMPLE_FACTOR = 2
f               = UPSAMPLE_FACTOR

PSF_HALFWIDTH   = 3
IBP_ITERATIONS  = 80
IBP_STEP_SIZE   = 0.5

# 4 diagonal corners and their nominal (dy, dx) shifts in sensor/LR pixels
CORNER_POSITIONS = [
    ('pos0', '(-x,+y)', (+0.5, -0.5)),
    ('pos2', '(+x,+y)', (+0.5, +0.5)),
    ('pos6', '(-x,-y)', (-0.5, -0.5)),
    ('pos8', '(+x,-y)', (-0.5, +0.5)),
]

COLORS = {'Native-2x': 'C0', 'SAA': 'C2', 'SAA+IBP': 'C3'}


# ── Image helpers ──────────────────────────────────────────────────────────────

def load_gray(path):
    img = np.array(Image.open(path), dtype=np.float64)
    return img.mean(axis=2) if img.ndim == 3 else img


def find_focus_dir(session_dir):
    """Return the dist+0.0mm_* sub-directory (best focus)."""
    for name in os.listdir(session_dir):
        if name.startswith('dist+0.0mm_') and os.path.isdir(
                os.path.join(session_dir, name)):
            return os.path.join(session_dir, name)
    raise FileNotFoundError(f'No dist+0.0mm_* folder in {session_dir}')


def find_rep_dirs(focus_dir):
    """Return sorted list of shift*_cycle*_rep* sub-directories."""
    dirs = sorted([
        os.path.join(focus_dir, d) for d in os.listdir(focus_dir)
        if d.startswith('shift') and os.path.isdir(os.path.join(focus_dir, d))
    ])
    if not dirs:
        raise FileNotFoundError(f'No shift* sub-dirs in {focus_dir}')
    return dirs


def find_pos_image(rep_dir, pos_prefix):
    """Find the image file for a given position prefix (e.g. 'pos0')."""
    for fname in os.listdir(rep_dir):
        if fname.startswith(pos_prefix + '_') and fname.endswith('.png'):
            return os.path.join(rep_dir, fname)
    raise FileNotFoundError(f'No image for {pos_prefix} in {rep_dir}')


def load_session(session_dir):
    """
    Load the 4 diagonal-corner LR frames for a session, per rep.

    Returns:
        all_reps     : list of N_reps lists, each containing 4 LR arrays
        shifts_lr    : list of 4 (dy_lr, dx_lr) tuples in sensor pixel units
        session_name : str
    """
    session_name = os.path.basename(session_dir)
    focus_dir    = find_focus_dir(session_dir)
    rep_dirs     = find_rep_dirs(focus_dir)

    print(f'  Focus dir : {os.path.basename(focus_dir)}')
    print(f'  Rep dirs  : {len(rep_dirs)} found')

    shifts_lr = [(dy, dx) for _, _, (dy, dx) in CORNER_POSITIONS]
    for pos_prefix, label, (dy, dx) in CORNER_POSITIONS:
        print(f'    {pos_prefix} {label:10s}: '
              f'shift_lr = (dy={dy:+.1f}, dx={dx:+.1f}) px')

    all_reps = []
    for rep_idx, rd in enumerate(rep_dirs):
        lr_frames = [load_gray(find_pos_image(rd, pos_prefix))
                     for pos_prefix, _, _ in CORNER_POSITIONS]
        all_reps.append(lr_frames)
        print(f'    rep {rep_idx}: loaded 4 frames from {os.path.basename(rd)}')

    return all_reps, shifts_lr, session_name


# ── PSF loading ────────────────────────────────────────────────────────────────

def load_psf():
    """
    Find all pos4_(0,0).png files under PSF_CALIB_DIR, align each by its
    peak location, average the aligned patches, then extract the PSF kernel.
    """
    MARGIN = PSF_HALFWIDTH + 6  # larger crop for alignment stability

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

    # Extract kernel from centre of averaged patch
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
    axes[0, 0].set_title('LR mean', fontsize=9)
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

    plt.suptitle(f'Mono SR — {title}', fontsize=10)
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


# ── Per-session processing ─────────────────────────────────────────────────────

def process_session(session_dir, psf_kernel):
    session_name = os.path.basename(session_dir)
    base_out_dir = os.path.join(session_dir, 'sr_output')

    print(f'\nLoading session: {session_name}')
    all_reps, shifts_lr, _ = load_session(session_dir)
    print(f'  {len(all_reps)} rep(s) to process')

    for rep_idx, lr_frames in enumerate(all_reps):
        out_dir   = os.path.join(base_out_dir, f'rep{rep_idx}')
        done_flag = os.path.join(out_dir, 'done.flag')
        os.makedirs(out_dir, exist_ok=True)

        if os.path.exists(done_flag):
            print(f'  [skip] rep {rep_idx} — already done')
            continue

        mean_lr = np.mean(lr_frames, axis=0)
        print(f'  Rep {rep_idx} — LR frame shape: {mean_lr.shape}')

        print('    Computing Native-2x ...')
        t0 = time.time()
        native_hr = ndi_zoom(mean_lr, f, order=3)
        print(f'    Done {time.time()-t0:.1f}s  shape={native_hr.shape}')

        print('    Computing SAA ...')
        t0 = time.time()
        saa_hr = shift_and_add(lr_frames, shifts_lr, factor=f, order=3)
        print(f'    Done {time.time()-t0:.1f}s')

        print(f'    Computing SAA+IBP ({IBP_ITERATIONS} iters) ...')
        t0 = time.time()
        ibp_hr, ibp_errors = ibp(
            lr_frames, shifts_lr, psf_kernel, saa_hr.copy(),
            factor=f, n_iter=IBP_ITERATIONS, step=IBP_STEP_SIZE,
        )
        print(f'    Done {time.time()-t0:.1f}s')

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
            print(f'    Saved {name_map[name]}.png')

        arr_lr = np.clip(mean_lr, 0, 255).astype(np.uint8)
        Image.fromarray(arr_lr).save(os.path.join(out_dir, 'LR_mean.png'))
        print('    Saved LR_mean.png')

        save_comparison(hr_images, mean_lr, out_dir, f'{session_name} rep{rep_idx}')
        save_convergence(ibp_errors, out_dir)
        open(done_flag, 'w').close()
        print(f'    Output: {out_dir}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    sessions = sorted([
        os.path.join(DATA_ROOT, d)
        for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ])
    if not sessions:
        raise FileNotFoundError(f'No session folders found in {DATA_ROOT}')

    print(f'Found {len(sessions)} session(s):\n' +
          '\n'.join(f'  {os.path.basename(s)}' for s in sessions))

    print('\nLoading PSF kernel ...')
    psf_kernel = load_psf()

    t_total = time.time()
    for i, session_dir in enumerate(sessions, 1):
        print(f'\n[{i}/{len(sessions)}] ', end='')
        process_session(session_dir, psf_kernel)

    print(f'\nAll sessions done in {(time.time()-t_total)/60:.1f} min')


if __name__ == '__main__':
    main()
