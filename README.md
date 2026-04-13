# Super-Resolution Experiments

Organized experiments for XPR mirror-based super-resolution imaging.
Each experiment folder follows a `data/` + `results/` convention where applicable.

## Hardware API

| File | Description |
|------|-------------|
| `api/daheng_camera.py` | Daheng camera wrapper (capture, exposure, HW trigger) |
| `api/xpr_controller.py` | OptoTune XPR mirror serial control |
| `api/allied_vision_camera.py` | Allied Vision camera wrapper (VmbPy SDK, burst streaming) |

## Calibration

| Folder | Scripts | What it does |
|--------|---------|--------------|
| `calibration_beam_shift/` | `calibrate_shift_grid.py` (collect), `plot_beam_shifts.py` (analyse) | Image a pinhole through a 3x3 grid of mirror positions at varying tilt angles. Measures pixel shift vs tilt. Data also used by `calibration_psf/`. |
| `calibration_psf/` | `analyze_psf_mtf.py` | PSF Gaussian fit + MTF analysis grouped by mirror position. Reads images from `calibration_beam_shift/data/`. |
| `calibration_autofocus/` | `calibrate_autofocus.py` (collect, GUI), `plot_depth_of_field.py` (analyse) | Sweep focus distance, measure Laplacian variance, plot usable depth of field. |
| `calibration_mech_stability/` | `rolling_stability.py` | Measure mechanical jitter of XPR mirror using Allied Vision camera burst captures. |

## Data Collection

| File | Description |
|------|-------------|
| `data_collection/collect_hw_triggered.py` | Collect raw images using hardware-triggered capture (XPR GPIO pulse) |
| `data_collection/collect_sw_triggered.py` | Collect raw images using software-triggered capture |
| `data_collection/psf_mtf_utils.py` | Shared utilities for PSF extraction and Gaussian fitting |

## SR Experiments

Each folder contains `run_sr.py` (runs Native-2x, SAA, SAA+IBP) and an `analysis.ipynb` notebook.

| Folder | Target | Sensor mode | LR size | HR size |
|--------|--------|-------------|---------|---------|
| `mono_cal_target/` | ISO 12233 chart | Mono | 1536x2048 | 3072x4096 |
| `rgb_cal_target/` | ISO 12233 chart | RGB (red Bayer channel) | 768x1024 | 1536x2048 |
| `mono_barcodes/` | Barcode sheets (2-6 mil) | Mono | 1536x2048 | 3072x4096 |
| `rgb_barcodes/` | Barcode sheets (2-6 mil) | RGB (red Bayer channel) | 768x1024 | 1536x2048 |

All `run_sr.py` scripts accept `--psf gaussian` (default) or `--psf measured` (loads pinhole PSF from `calibration_beam_shift/data/`).
