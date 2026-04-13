#!/usr/bin/env python3
"""
sweep_sr.py  --  RGB calibration-target super resolution

Processes RGB (Bayer RGGB) calibration target images.

Data layout
-----------
  data/<session>/<combo>/
      corner{0-3}_rep{00-04}.png   (5 repeats of each of 4 shift positions)
      metadata.json                (expected shifts in sensor pixels)

Red channel extraction
----------------------
Red pixels: img[0::2, 0::2]  ->  768 x 1024 LR pixels
1 sensor pixel shift = 0.5 red-channel LR pixel
Upsample factor 2  ->  1536 x 2048 HR (= full sensor resolution)

SR methods
----------
  Native-2x  : bicubic 2x upsample of the average of all 4 corners
  SAA        : Shift-and-Add using expected shifts from metadata
  SAA+IBP    : IBP initialised from SAA

Usage
-----
  python sweep_sr.py                          # Gaussian PSF (default)
  python sweep_sr.py --psf measured           # Measured PSF from calibration
  python sweep_sr.py --psf measured --psf-dir /path/to/psf_images
"""

import argparse
import json
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

# -- Sensor / Bayer --------------------------------------------------------

BAYER_ROW_OFFSET = 0   # red pixels at even rows
BAYER_COL_OFFSET = 0   # red pixels at even cols
UPSAMPLE_FACTOR  = 2
f                = UPSAMPLE_FACTOR

# -- SR parameters ---------------------------------------------------------

PSF_SIZE       = 7
PSF_SIGMA      = 1.0
PSF_HALFWIDTH  = 3
IBP_ITERATIONS = 50
IBP_STEP_SIZE  = 0.5

CORNER_ORDER = ['(-x,+y)', '(+x,+y)', '(-x,-y)', '(+x,-y)']
COLORS  = {'Native-2x': 'C0', 'SAA': 'C2', 'SAA+IBP': 'C3'}


# -- Image helpers ---------------------------------------------------------

def load_gray(path):
    img = np.array(Image.open(path), dtype=np.float64)
    return img.mean(axis=2) if img.ndim == 3 else img


def extract_red(img):
    """Extract the red Bayer channel (even rows, even cols) from RGGB image."""
    return img[BAYER_ROW_OFFSET::2, BAYER_COL_OFFSET::2].copy()


def load_combo(combo_dir):
    """Load a tilt/settle combo directory.

    Returns:
        lr_frames  : list of 4 red-channel LR arrays (averaged over reps)
        shifts_lr  : list of 4 (dy_lr, dx_lr) tuples in red-channel pixel units
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
            f for f in os.listdir(combo_dir)
            if f.startswith(f'corner{idx}_rep') and f.endswith('.png')
        ])
        if not reps:
            raise FileNotFoundError(f'No images for corner{idx} in {combo_dir}')
        stack = np.stack([extract_red(load_gray(os.path.join(combo_dir, r))) for r in reps])
        lr_frames.append(stack.mean(axis=0))
        shifts_lr.append(get_shift(label))
        print(f'    corner{idx} {label:10s}: shift_lr=(dy={shifts_lr[-1][0]:+.3f}, '
              f'dx={shifts_lr[-1][1]:+.3f}) LR px  ({len(reps)} reps)')

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


def load_measured_psf(psf_dir):
    """Load measured PSF by averaging pos4_(0,0).png files across sweep dirs."""
    MARGIN = PSF_HALFWIDTH + 6

    patches = []
    for sweep_dir in sorted(os.listdir(psf_dir)):
        full = os.path.join(psf_dir, sweep_dir)
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
        raise FileNotFoundError(f'No pos4_(0,0).png found under {psf_dir}')

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
        up = ndi_zoom(lr, factor, order=order)
        acc += ndi_shift(up, (dy * factor, dx * factor), order=3, mode='nearest')
    return acc / len(lr_list)


def ibp(lr_list, shifts_yx, kernel, hr_init, factor=2, n_iter=50, step=0.5):
    hr = hr_init.copy()
    n  = len(lr_list)
    errors = []
    for it in range(n_iter):
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
        if (it + 1) % 10 == 0:
            print(f'    iter {it+1:3d}/{n_iter}  MSE = {errors[-1]:.4f}')
    return hr, errors


# -- Plotting --------------------------------------------------------------

def save_comparison(hr_images, mean_lr, out_dir, title):
    H, W = list(hr_images.values())[0].shape
    cr = slice(H // 2 - 100, H // 2 + 100)
    cc = slice(W // 2 - 100, W // 2 + 100)

    n = len(hr_images) + 1
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
    axes[1, 0].set_title('LR bicubic 2x (mean)', fontsize=8)
    axes[1, 0].axis('off')
    for i, (name, img) in enumerate(hr_images.items(), 1):
        axes[1, i].imshow(img[cr, cc], cmap='gray', interpolation='nearest')
        axes[1, i].set_title(name, fontsize=8)
        axes[1, i].axis('off')

    plt.suptitle(f'Red channel SR - {title}', fontsize=10)
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


# -- Per-combo processing --------------------------------------------------

def process_combo(combo_dir, psf_kernel, output_base):
    combo_name = os.path.basename(combo_dir)
    out_dir    = os.path.join(output_base, combo_name)
    os.makedirs(out_dir, exist_ok=True)
    done_flag  = os.path.join(out_dir, 'done.flag')

    if os.path.exists(done_flag):
        print(f'  [skip] {combo_name} - already done')
        return

    print(f'\nLoading combo: {combo_name}')
    lr_frames, shifts_lr = load_combo(combo_dir)
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

    with open(os.path.join(out_dir, 'shifts.json'), 'w') as fp:
        json.dump({'shifts_lr_yx': shifts_lr, 'corner_labels': CORNER_ORDER}, fp, indent=2)

    save_comparison(hr_images, mean_lr, out_dir, combo_name)
    save_convergence(ibp_errors, out_dir)
    open(done_flag, 'w').close()
    print(f'  Output: {out_dir}')


# -- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Super-resolution for RGB calibration target images (Bayer RGGB)')
    parser.add_argument('--psf', choices=['gaussian', 'measured'], default='gaussian',
                        help='PSF type: gaussian (default) or measured from calibration data')
    parser.add_argument('--psf-dir', default=None,
                        help='Path to PSF calibration images (default: ../calibration_beam_shift/data/)')
    parser.add_argument('--data-dir', default=None,
                        help='Path to data directory (default: ./data/)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: ./results/)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = args.data_dir or os.path.join(script_dir, 'data')
    output_dir = args.output_dir or os.path.join(script_dir, 'results')
    psf_dir    = args.psf_dir or os.path.join(script_dir, '..', 'calibration_beam_shift', 'data')

    # Find all session directories (contain metadata.json)
    combos = []
    for name in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'metadata.json')):
            combos.append(d)

    if not combos:
        raise FileNotFoundError(f'No directories with metadata.json found under {data_dir}')

    print(f'Found {len(combos)} session(s):\n' +
          '\n'.join(f'  {os.path.basename(c)}' for c in combos))

    if args.psf == 'measured':
        print('\nLoading measured PSF kernel ...')
        psf_kernel = load_measured_psf(psf_dir)
    else:
        print('\nCreating Gaussian PSF kernel ...')
        psf_kernel = make_gaussian_psf()

    t_total = time.time()
    for i, combo_dir in enumerate(combos, 1):
        print(f'\n[{i}/{len(combos)}] ', end='')
        process_combo(combo_dir, psf_kernel, output_dir)

    print(f'\nAll combos done in {(time.time()-t_total)/60:.1f} min')


if __name__ == '__main__':
    main()
