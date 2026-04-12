#!/usr/bin/env python3
"""
sweep_sr_barcodes_mono_cal.py

Like sweep_sr_barcodes_mono.py but uses CALIBRATED per-corner shifts instead
of nominal ±0.5 px.  Shifts are obtained from:

    imaging/calibration/psf_images/shifts.csv

using superposition of the x-sweep and y-sweep calibration data:

    dx_cal = dx(x-sweep, pos, |tilt_x_deg|) + dx(y-sweep, pos, |tilt_y_deg|)
    dy_cal = dy(x-sweep, pos, |tilt_x_deg|) + dy(y-sweep, pos, |tilt_y_deg|)

All angles are interpolated linearly.  The calibrated shifts replace the
hardcoded ±0.5 px nominal values.

Output goes to  <session>/sr_output_cal/  (distinct from sr_output/).
"""

import os
import csv
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

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT     = os.path.join(os.path.dirname(SCRIPT_DIR), 'mono_barcodes')
PSF_IMAGE_PATH = os.path.join(
    os.path.dirname(SCRIPT_DIR), 'calibration', 'psf_images',
    'sweepx_tilt0.30000deg', 'pos4_(0,0).png'
)
CAL_CSV_PATH  = os.path.join(
    os.path.dirname(SCRIPT_DIR), 'calibration', 'psf_images', 'shifts.csv'
)

# ── SR parameters ──────────────────────────────────────────────────────────────

UPSAMPLE_FACTOR = 2
f               = UPSAMPLE_FACTOR

PSF_HALFWIDTH   = 3
IBP_ITERATIONS  = 80
IBP_STEP_SIZE   = 0.5

# 4 diagonal corners used to index the calibration CSV
CORNER_POSITIONS = [
    ('pos0', '(-x,+y)'),
    ('pos2', '(+x,+y)'),
    ('pos6', '(-x,-y)'),
    ('pos8', '(+x,-y)'),
]
CORNER_IDX = {label: int(label[3:]) for label, _ in CORNER_POSITIONS}  # 'pos0'→0

COLORS = {'Native-2x': 'C0', 'SAA': 'C2', 'SAA+IBP': 'C3'}

OUTPUT_SUBDIR = 'sr_output_cal'


# ── Calibration CSV ─────────────────────────────────────────────────────────────

def load_cal_csv(path):
    """
    Return a dict:
      cal[sweep_axis][pos_idx] = {'tilt': [...], 'dx': [...], 'dy': [...]}
    Sorted by tilt_angle_deg for use with np.interp.
    """
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'axis':  row['sweep_axis'],
                'tilt':  float(row['tilt_angle_deg']),
                'pos':   int(row['position']),
                'dx':    float(row['dx_mean_px']),
                'dy':    float(row['dy_mean_px']),
            })

    cal = {'x': {}, 'y': {}}
    for axis in ('x', 'y'):
        axis_rows = [r for r in rows if r['axis'] == axis]
        positions = sorted(set(r['pos'] for r in axis_rows))
        for pos in positions:
            pr = sorted([r for r in axis_rows if r['pos'] == pos],
                        key=lambda r: r['tilt'])
            cal[axis][pos] = {
                'tilt': np.array([r['tilt'] for r in pr]),
                'dx':   np.array([r['dx']   for r in pr]),
                'dy':   np.array([r['dy']   for r in pr]),
            }
    return cal


def interp_shift(cal, axis, pos_idx, tilt_abs):
    """Interpolate (dx, dy) from calibration for given axis, position, tilt magnitude."""
    entry = cal[axis][pos_idx]
    dx = float(np.interp(tilt_abs, entry['tilt'], entry['dx']))
    dy = float(np.interp(tilt_abs, entry['tilt'], entry['dy']))
    return dx, dy


def get_calibrated_shifts(focus_dir, cal):
    """
    Read metadata.json from the first rep dir in focus_dir.
    Return list of (dy_cal, dx_cal) in sensor/LR pixels for the 4 diagonal corners.
    """
    rep_dirs = find_rep_dirs(focus_dir)
    meta_path = os.path.join(rep_dirs[0], 'metadata.json')
    with open(meta_path) as mf:
        meta = json.load(mf)

    tilt_x_abs = abs(meta['tilt_x_deg'])
    tilt_y_abs = abs(meta['tilt_y_deg'])

    print(f'  Calibration: tilt_x_abs={tilt_x_abs:.5f} deg, '
          f'tilt_y_abs={tilt_y_abs:.5f} deg')

    shifts = []
    for pos_prefix, pos_label in CORNER_POSITIONS:
        pos_idx = int(pos_prefix[3:])  # 'pos0' → 0

        dx_x, dy_x = interp_shift(cal, 'x', pos_idx, tilt_x_abs)
        dx_y, dy_y = interp_shift(cal, 'y', pos_idx, tilt_y_abs)

        dx_cal = dx_x + dx_y
        dy_cal = dy_x + dy_y

        shifts.append((dy_cal, dx_cal))
        print(f'    {pos_prefix} {pos_label:10s}: '
              f'dx_cal={dx_cal:+.4f}  dy_cal={dy_cal:+.4f} px '
              f'(x: dx={dx_x:+.4f} dy={dy_x:+.4f}  '
              f'y: dx={dx_y:+.4f} dy={dy_y:+.4f})')

    return shifts


# ── Image helpers ──────────────────────────────────────────────────────────────

def load_gray(path):
    img = np.array(Image.open(path), dtype=np.float64)
    return img.mean(axis=2) if img.ndim == 3 else img


def find_focus_dir(session_dir):
    for name in os.listdir(session_dir):
        if name.startswith('dist+0.0mm_') and os.path.isdir(
                os.path.join(session_dir, name)):
            return os.path.join(session_dir, name)
    raise FileNotFoundError(f'No dist+0.0mm_* folder in {session_dir}')


def find_rep_dirs(focus_dir):
    dirs = sorted([
        os.path.join(focus_dir, d) for d in os.listdir(focus_dir)
        if d.startswith('shift') and os.path.isdir(os.path.join(focus_dir, d))
    ])
    if not dirs:
        raise FileNotFoundError(f'No shift* sub-dirs in {focus_dir}')
    return dirs


def find_pos_image(rep_dir, pos_prefix):
    for fname in os.listdir(rep_dir):
        if fname.startswith(pos_prefix + '_') and fname.endswith('.png'):
            return os.path.join(rep_dir, fname)
    raise FileNotFoundError(f'No image for {pos_prefix} in {rep_dir}')


def load_session(session_dir, cal):
    """
    Load the 4 diagonal-corner LR frames per rep and compute calibrated shifts.

    Returns:
        all_reps     : list of N_reps lists, each containing 4 LR arrays
        shifts_cal   : list of 4 (dy_cal, dx_cal) in sensor/LR pixel units
        session_name : str
    """
    session_name = os.path.basename(session_dir)
    focus_dir    = find_focus_dir(session_dir)
    rep_dirs     = find_rep_dirs(focus_dir)

    print(f'  Focus dir : {os.path.basename(focus_dir)}')
    print(f'  Rep dirs  : {len(rep_dirs)} found')

    shifts_cal = get_calibrated_shifts(focus_dir, cal)

    all_reps = []
    for rep_idx, rd in enumerate(rep_dirs):
        lr_frames = [load_gray(find_pos_image(rd, pos_prefix))
                     for pos_prefix, _ in CORNER_POSITIONS]
        all_reps.append(lr_frames)
        print(f'    rep {rep_idx}: loaded 4 frames from {os.path.basename(rd)}')

    return all_reps, shifts_cal, session_name


# ── PSF loading ────────────────────────────────────────────────────────────────

def load_psf():
    if not os.path.exists(PSF_IMAGE_PATH):
        sigma = 1.0
        print(f'  PSF image not found — using Gaussian (sigma={sigma} px)')
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

    plt.suptitle(f'Mono SR (cal) — {title}', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison.png'), bbox_inches='tight', dpi=100)
    plt.close(fig)
    print('  Saved comparison.png')


def save_convergence(ibp_errors, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ibp_errors, color=COLORS['SAA+IBP'], lw=1.5)
    ax.set_title('IBP convergence (calibrated shifts)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'convergence.png'), bbox_inches='tight')
    plt.close(fig)
    print('  Saved convergence.png')


# ── Per-session processing ─────────────────────────────────────────────────────

def process_session(session_dir, psf_kernel, cal):
    session_name = os.path.basename(session_dir)
    base_out_dir = os.path.join(session_dir, OUTPUT_SUBDIR)

    print(f'\nLoading session: {session_name}')
    all_reps, shifts_cal, _ = load_session(session_dir, cal)
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

        print('    Computing SAA (calibrated) ...')
        t0 = time.time()
        saa_hr = shift_and_add(lr_frames, shifts_cal, factor=f, order=3)
        print(f'    Done {time.time()-t0:.1f}s')

        print(f'    Computing SAA+IBP ({IBP_ITERATIONS} iters, calibrated) ...')
        t0 = time.time()
        ibp_hr, ibp_errors = ibp(
            lr_frames, shifts_cal, psf_kernel, saa_hr.copy(),
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

    print(f'\nLoading calibration CSV: {CAL_CSV_PATH}')
    cal = load_cal_csv(CAL_CSV_PATH)
    print(f'  Loaded. Axes: {list(cal.keys())}, '
          f'positions: {sorted(cal["x"].keys())}')

    print('\nLoading PSF kernel ...')
    psf_kernel = load_psf()

    t_total = time.time()
    for i, session_dir in enumerate(sessions, 1):
        print(f'\n[{i}/{len(sessions)}] ', end='')
        process_session(session_dir, psf_kernel, cal)

    print(f'\nAll sessions done in {(time.time()-t_total)/60:.1f} min')


if __name__ == '__main__':
    main()
