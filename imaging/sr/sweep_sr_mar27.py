#!/usr/bin/env python3
"""
sweep_sr_mar27.py

Super-resolution sweep for the March 27 RGB (Bayer) image datasets.

Data layout
-----------
Two session folders inside mar27_rgb_images/:
  20260327_125058  →  settle 5 ms, 20 ms
  20260327_125302  →  settle 50 ms, 500 ms

Each session has sub-folders like tilt0.36000deg_settle20ms/ containing:
  corner{0-3}_rep{00-04}.png   (5 repeats of each of 4 shift positions)
  metadata.json                (expected shifts in sensor pixels)

Special folders (special_1.0px_settle*ms) use per-corner optimised tilts to
achieve exactly 1.0 sensor-pixel shifts; these are processed identically.

Red channel extraction
----------------------
The images are stored as raw Bayer grayscale (RGGB pattern).
Red pixels: img[0::2, 0::2]  →  768 × 1024 LR pixels
1 sensor pixel shift  ≡  0.5 red-channel LR pixel
Upsample factor 2  →  1536 × 2048 HR (= full sensor resolution)

Shift convention
----------------
Shifts from metadata are in sensor pixels.  Divide by 2 for red-channel LR
space.  Corner positions are labelled (-x,+y), (+x,+y), (-x,-y), (+x,-y);
expected_dx_px / expected_dy_px give the shift of THAT frame relative to the
nominal centre (0,0).

SR methods
----------
  Native-2x  : bicubic 2× upsample of the average of all 4 corners (blurred
               but centred; serves as the no-SR baseline at HR resolution)
  SAA        : Shift-and-Add using expected shifts from metadata
  SAA+IBP    : IBP initialised from SAA
  TV-SR      : Total-Variation SR initialised from SAA
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
DATA_ROOT     = os.path.join(SCRIPT_DIR, 'mar27_rgb_images')
SESSION_DIRS  = ['20260327_125058', '20260327_125302']
PSF_CALIB_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'calibration', '20260326_152815')

# ── Sensor / Bayer ─────────────────────────────────────────────────────────────

BAYER_ROW_OFFSET = 0   # red pixels start at even rows
BAYER_COL_OFFSET = 0   # red pixels start at even cols
UPSAMPLE_FACTOR  = 2
f                = UPSAMPLE_FACTOR

PIXEL_PITCH_UM   = 3.45                      # sensor pixel pitch µm
RED_PITCH_LR_UM  = PIXEL_PITCH_UM * 2        # red channel LR pixel pitch
PITCH_HR_UM      = PIXEL_PITCH_UM            # 1 HR pixel = 1 sensor pixel
NYQUIST_LR_CYMM  = 0.5 / (RED_PITCH_LR_UM * 1e-3)   # 72.5 cy/mm
NYQUIST_HR_CYMM  = 0.5 / (PITCH_HR_UM  * 1e-3)       # 144.9 cy/mm

# ── IBP parameters ─────────────────────────────────────────────────────────────

PSF_HALFWIDTH  = 3
IBP_ITERATIONS = 10
IBP_STEP_SIZE  = 0.5

# Per-combo iteration overrides — (session, combo) → n_iter
IBP_ITER_OVERRIDES = {
    ('20260327_125058', 'special_1.0px_settle20ms'): 50,
}

# IBP is only run on a crop of the HR image to keep runtime manageable.
# Set to None to run on the full image (slow). Once ROIs are defined in the
# notebook, set this to the specific region of interest instead.
IBP_CROP_HR = None  # None = full image IBP (no crop artifact)

COLORS  = {'Native-2x': 'C0', 'SAA': 'C2', 'SAA+IBP': 'C3'}
METHODS = ['Native-2x', 'SAA', 'SAA+IBP']

# ── Corner label → shift sign convention ──────────────────────────────────────

CORNER_ORDER = ['(-x,+y)', '(+x,+y)', '(-x,-y)', '(+x,-y)']


# ── Image helpers ──────────────────────────────────────────────────────────────

def load_gray(path):
    img = np.array(Image.open(path), dtype=np.float64)
    return img.mean(axis=2) if img.ndim == 3 else img


def extract_red(img):
    """Extract the red Bayer channel (even rows, even cols)."""
    return img[BAYER_ROW_OFFSET::2, BAYER_COL_OFFSET::2].copy()


def load_combo(combo_dir):
    """
    Load a tilt/settle combo directory.
    Returns:
        lr_frames  : list of 4 red-channel LR arrays (averaged over reps)
        shifts_lr  : list of 4 (dy_lr, dx_lr) tuples in red-channel pixel units
        meta       : parsed metadata dict
    """
    with open(os.path.join(combo_dir, 'metadata.json')) as fp:
        meta = json.load(fp)

    # Build expected shifts lookup (sensor pixels → red LR pixels)
    # metadata may use 'expected_shifts' dict or per-corner entries
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

    return lr_frames, shifts_lr, meta


# ── SR core functions ──────────────────────────────────────────────────────────

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


def ibp(lr_list, shifts_yx, kernel, hr_init, factor=2, n_iter=20, step=0.5):
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




# ── PSF loading ────────────────────────────────────────────────────────────────

def load_psf():
    """
    Load PSF from pinhole calibration (pos4 = unshifted reference).
    For the red channel SR, HR pixels = sensor pixels, so the PSF kernel
    is at sensor resolution — no additional upsampling needed.
    """
    pos4_imgs = []
    for sweep_dir in sorted(os.listdir(PSF_CALIB_DIR)):
        full = os.path.join(PSF_CALIB_DIR, sweep_dir)
        if not os.path.isdir(full):
            continue
        pos4_path = os.path.join(full, 'pos4_(0,0).png')
        if os.path.exists(pos4_path):
            img = load_gray(pos4_path)
            pos4_imgs.append(img)

    if not pos4_imgs:
        raise FileNotFoundError(f'No pos4_(0,0).png found under {PSF_CALIB_DIR}')

    avg  = np.mean(pos4_imgs, axis=0)
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

    # No upsampling: 1 HR pixel = 1 sensor pixel for the red channel case
    print(f'  PSF: averaged {len(pos4_imgs)} pos4 images, kernel shape {kernel.shape}')
    return kernel


# ── Per-combo processing ───────────────────────────────────────────────────────

def process_combo(combo_dir, session_out_dir, psf_kernel, n_iter=IBP_ITERATIONS):
    combo_name = os.path.basename(combo_dir)
    out_dir    = os.path.join(session_out_dir, combo_name)
    os.makedirs(out_dir, exist_ok=True)
    done_flag  = os.path.join(out_dir, 'done.flag')

    if os.path.exists(done_flag):
        print('    [skip] already done')
        return

    lr_frames, shifts_lr, meta = load_combo(combo_dir)

    # Native-2x: bicubic upsample of the mean of all 4 LR frames
    mean_lr   = np.mean(lr_frames, axis=0)
    native_hr = ndi_zoom(mean_lr, f, order=3)

    # SAA (full image)
    saa_hr = shift_and_add(lr_frames, shifts_lr, f, order=3)

    # SAA + IBP — run only on a crop to keep runtime fast.
    # IBP_CROP_HR defines the HR crop size; crop is centred on the full image.
    H_hr, W_hr = saa_hr.shape
    if IBP_CROP_HR is not None:
        ch = min(IBP_CROP_HR, H_hr)
        cw = min(IBP_CROP_HR, W_hr)
        r0c = (H_hr - ch) // 2;  r1c = r0c + ch
        c0c = (W_hr - cw) // 2;  c1c = c0c + cw
        r0l, r1l = r0c // f, r1c // f
        c0l, c1l = c0c // f, c1c // f
        lr_crop   = [lr[r0l:r1l, c0l:c1l] for lr in lr_frames]
        saa_crop  = saa_hr[r0c:r1c, c0c:c1c]
        ibp_crop_region = (r0c, r1c, c0c, c1c)
    else:
        lr_crop  = lr_frames
        saa_crop = saa_hr
        ibp_crop_region = (0, H_hr, 0, W_hr)

    print('    IBP  ...', end=' ', flush=True)
    t0 = time.time()
    ibp_result, ibp_errors = ibp(lr_crop, shifts_lr, psf_kernel, saa_crop.copy(),
                                 factor=f, n_iter=n_iter, step=IBP_STEP_SIZE)
    print(f'{time.time()-t0:.0f}s')

    # Build full-size IBP output (SAA everywhere, IBP in the crop region)
    ibp_hr = saa_hr.copy()
    r0c, r1c, c0c, c1c = ibp_crop_region
    ibp_hr[r0c:r1c, c0c:c1c] = ibp_result

    hr_images = {
        'Native-2x': native_hr,
        'SAA':       saa_hr,
        'SAA+IBP':   ibp_hr,
    }

    # Save each HR image as PNG
    for name, img in hr_images.items():
        arr = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(out_dir, f'{name.replace("+","_")}.png'))

    # Also save the mean LR red frame at native resolution for reference
    arr_lr = np.clip(mean_lr, 0, 255).astype(np.uint8)
    Image.fromarray(arr_lr).save(os.path.join(out_dir, 'LR_red_mean.png'))

    # Save shifts used
    with open(os.path.join(out_dir, 'shifts.json'), 'w') as fp:
        json.dump({'shifts_lr_yx': shifts_lr, 'corner_labels': CORNER_ORDER}, fp, indent=2)

    # Comparison figure: all methods on a representative central crop
    _save_comparison_plot(hr_images, mean_lr, out_dir)
    _save_convergence_plot(ibp_errors, out_dir)

    open(done_flag, 'w').close()


def _save_comparison_plot(hr_images, lr_ref, out_dir):
    """Full-image comparison strip + a zoomed central crop."""
    H, W = list(hr_images.values())[0].shape
    # Central crop: 200×200 HR pixels
    cr = slice(H // 2 - 100, H // 2 + 100)
    cc = slice(W // 2 - 100, W // 2 + 100)

    n = len(hr_images) + 1   # +1 for LR reference
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    # Row 0: full image (downsampled for display)
    ds = 4   # display downscale
    lr_disp = ndi_zoom(lr_ref, 1, order=1)
    axes[0, 0].imshow(lr_disp[::ds, ::ds], cmap='gray', interpolation='nearest')
    axes[0, 0].set_title('LR red (avg reps)', fontsize=9)
    axes[0, 0].axis('off')
    for i, (name, img) in enumerate(hr_images.items(), 1):
        axes[0, i].imshow(img[::ds*2, ::ds*2], cmap='gray', interpolation='nearest')
        axes[0, i].set_title(name, fontsize=9)
        axes[0, i].axis('off')

    # Row 1: zoomed central crop
    lr_zoom = ndi_zoom(lr_ref, 2, order=3)
    axes[1, 0].imshow(lr_zoom[cr, cc], cmap='gray', interpolation='nearest')
    axes[1, 0].set_title('LR bicubic 2× (single frame)', fontsize=8)
    axes[1, 0].axis('off')
    for i, (name, img) in enumerate(hr_images.items(), 1):
        axes[1, i].imshow(img[cr, cc], cmap='gray', interpolation='nearest')
        axes[1, i].set_title(name, fontsize=8)
        axes[1, i].axis('off')

    plt.suptitle(f'Red channel SR — {os.path.basename(out_dir)}', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison.png'), bbox_inches='tight', dpi=100)
    plt.close(fig)


def _save_convergence_plot(ibp_errors, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ibp_errors, color=COLORS['SAA+IBP'], lw=1.5)
    ax.set_title('IBP convergence'); ax.set_xlabel('Iter')
    ax.set_ylabel('MSE'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'convergence.png'), bbox_inches='tight')
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('Loading PSF kernel ...')
    psf_kernel = load_psf()

    combos_total = []
    for session in SESSION_DIRS:
        session_dir = os.path.join(DATA_ROOT, session)
        if not os.path.isdir(session_dir):
            print(f'  [skip] {session} not found')
            continue
        subdirs = sorted([
            d for d in os.listdir(session_dir)
            if os.path.isdir(os.path.join(session_dir, d))
            and os.path.exists(os.path.join(session_dir, d, 'metadata.json'))
        ])
        for d in subdirs:
            combos_total.append((session, d))

    print(f'Found {len(combos_total)} combos across {len(SESSION_DIRS)} sessions\n')

    t_total = time.time()
    for i, (session, combo_name) in enumerate(combos_total):
        combo_dir      = os.path.join(DATA_ROOT, session, combo_name)
        session_out    = os.path.join(DATA_ROOT, session, 'results')
        os.makedirs(session_out, exist_ok=True)

        elapsed = time.time() - t_total
        eta_str = ''
        if i > 0:
            eta = elapsed / i * (len(combos_total) - i)
            eta_str = f'  ETA {eta/60:.1f} min'

        n_iter = IBP_ITER_OVERRIDES.get((session, combo_name), IBP_ITERATIONS)

        # If this combo has an override, skip all others
        if IBP_ITER_OVERRIDES and (session, combo_name) not in IBP_ITER_OVERRIDES:
            continue

        # Remove done.flag so overridden combos are always reprocessed
        done_flag = os.path.join(session_out, combo_name, 'done.flag')
        if os.path.exists(done_flag) and (session, combo_name) in IBP_ITER_OVERRIDES:
            os.remove(done_flag)

        print(f'[{i+1:2d}/{len(combos_total)}] {session}/{combo_name}  (n_iter={n_iter}){eta_str}')
        try:
            process_combo(combo_dir, session_out, psf_kernel, n_iter=n_iter)
        except Exception as e:
            print(f'    ERROR: {e}')

    print(f'\nDone in {(time.time()-t_total)/60:.1f} min')
    print(f'Results saved under mar27_rgb_images/{{session}}/results/')


if __name__ == '__main__':
    main()
