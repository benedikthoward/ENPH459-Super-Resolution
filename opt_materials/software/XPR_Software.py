import logging
import math
import cv2
import time
import optoICC
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
import gxipy as gx # Daheng SDK
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from XPR_Window import *
from pypylon import pylon# Basler
from optoControllerToolbox.SmartFilter import SmartFilters


import ctypes


# This is done to use the XPR logo as taskbar image (Windows only)
if sys.platform == 'win32':
    myappid = 'string'  # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

# XPR init
# Restarting step (as to be done due to filter set step, need a power cycle)
icc4c = optoICC.connect()
icc4c.reset(force=True)
icc4c.go_pro()

connected_devices = [optoICC.DeviceModel(icc4c.MiscFeatures.GetDeviceType(0)),
                     optoICC.DeviceModel(icc4c.MiscFeatures.GetDeviceType(1)),
                     optoICC.DeviceModel(icc4c.MiscFeatures.GetDeviceType(2)),
                     optoICC.DeviceModel(icc4c.MiscFeatures.GetDeviceType(3))]

ch_0 = icc4c.channel[0]
ch_0.StaticInput.SetAsInput()  # (1) here we tell the Manager that we will use a static input
ch_0.InputConditioning.SetGain(1.0)  # (2) here we tell the Manager some input conditioning parameters
ch_0.SetControlMode(optoICC.UnitType.UNITLESS)  # (3) here we tell the Manager that our input will be in units of degrees
# ch_0.LinearOutput.SetCurrentLimit(0.6)  #(4) here we tell the Manager to limit the current to 600mA (default)
#ch_0.Manager.CheckSignalFlow()  # This is a useful method to make sure the signal flow is configured correctly.

ch_1 = icc4c.channel[1]
ch_1.StaticInput.SetAsInput()  # (1) here we tell the Manager that we will use a static input
ch_1.InputConditioning.SetGain(1.0)  # (2) here we tell the Manager some input conditioning parameters
ch_1.SetControlMode(optoICC.UnitType.UNITLESS)  # (3) here we tell the Manager that our input will be in units of degrees
# ch_1.LinearOutput.SetCurrentLimit(0.6)  #(4) here we tell the Manager to limit the current to 600mA (default)
#ch_1.Manager.CheckSignalFlow()  # This is a useful method to make sure the signal flow is configured correctly.

si_0 = icc4c.channel[0].StaticInput
si_1 = icc4c.channel[1].StaticInput

si_0.SetValue(0)
si_1.SetValue(0)

# configure smart filters on both channels (axes)
smart_filters = SmartFilters(icc4c)
smart_filters.transition_time = 1.5e-3      # in seconds, maximum is 1.6ms
# CANNOT adjust sampling time, should be doable
smart_filters.channels = [0, 1]
smart_filters.configure_filters()

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Parameters and global variables
# Marker red line of the ROI window
w_line = 10
grab_mode = False
XPR_on = False
frame_CamImage = QImage()
frame_CamZoom = QImage()
frame_CamXPR = QImage()

frame_CamXPR_numpy = np.zeros((1, 1, 1))
FPS = 0
FPS2 = 0
FPS_n = 0.1
mutex = QMutex()
cameraType = 0  # 0: DAHENG Color Camera, 1: BASLER Mono Camera, 2: DAHENG Mono Camera
DAHENG = 0
BASLER = 1
DAHENG_MONO = 2
frame_number = 0
RGB = 0  # 0:RGB - 1:R - 2:G - 3:B
R = 1
G = 2
B = 3
channel = RGB
NORMAL = 0
INTERPOLATED = 1
interpolation_mode = NORMAL

WHITE = np.array([255, 255, 255])  # no comparison
RED = np.array([255, 0, 0])
BLUE = np.array([0, 0, 255])
REDBLUE = np.array([255, 0, 255])
color_comparison = WHITE
min_color_similarity = 145  # 151
max_val = 1 / math.sqrt(2)

save_enabled = False
save_dir = None
save_set_ts = ""
current_exposure = 50000
current_gain = 0

gt_original = None
gt_crop = None
gt_source_name = ""
snr_enabled = False
snr_xpr_psnr = float('nan')
snr_xpr_ssim = float('nan')
snr_sub_psnr = [float('nan')] * 4   # PSNR for each of the 4 raw sub-images
snr_sub_ssim = [float('nan')] * 4
snr_avg_psnr = float('nan')          # PSNR of the pixel-averaged sub-image
snr_avg_ssim = float('nan')
snr_sub_ecc = [float('nan')] * 4    # ECC correlation per sub-image
snr_avg_ecc = float('nan')
snr_xpr_ecc = float('nan')

_reg_data = None  # {"H_roi": 3×3 homography, "gt_rot": oriented GT, "ncc": float, "method": str, "src_h": int, "src_w": int}
_reg_status = ""       # "SIFT (ncc=X.XXX, N inliers)" or "Fallback"
_reg_debug = ""        # diagnostic text for GUI
_reg_match_img = None  # float32 H×W [0,1] grayscale or uint8 H×W×3 RGB (SIFT matches)
_sift_full_frame = None  # full-frame grayscale float32 [0,1], captured for SIFT registration

# Camera detection
try:
    # Try to detect BASLER camera
    cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    cam.Open()
    cam.ExposureAuto.SetValue("Off")
    cam.ExposureTime.SetValue(50000)
    cam.Gain.SetValue(0)

    # Select the Frame Start Trigger
    cam.TriggerSelector.SetValue("FrameStart")
    # Enable triggered image acquisition for the Frame Start trigger
    cam.TriggerMode.SetValue("On")
    # Set the trigger source for the Frame Start trigger to Software
    cam.TriggerSource.SetValue("Software")

    cameraType = BASLER
    h = 3648
    w = 5472
    tilt_angle = 0.05005  # dist(-tilt_angle to +tilt_angle) = 1/2px
except:
    # Try to detect DAHENG camera
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num == 0:
        print("No camera detected")
        sys.exit(1)
    elif dev_num > 0:
        print("Camera detected")

    for id_cam in range(dev_num):
        # Open device
        # Get the list of basic device information
        strSN = dev_info_list[id_cam].get("sn")
        # Open the device by serial number
        cam = device_manager.open_device_by_sn(strSN)
        # cam.data_stream[0].set_acquisition_buffer_number(1)
        # cam.data_stream[0].flush_queue()

        cam.data_stream[0].StreamBufferHandlingMode.set(3)
        cam.TriggerMode.set(1)
        # cam.data_stream[0].AcquisitionMode.set(0)
        cam.Gain.set(0)
        # Start acquisition
        cam.stream_on()

    # Detect mono vs colour by checking pixel colour filter.
    # On mono cameras PixelColorFilter is not readable — treat that as mono.
    try:
        is_colour = cam.PixelColorFilter.get() != gx.GxPixelColorFilterEntry.NONE
    except Exception:
        is_colour = False
    if is_colour:
        cameraType = DAHENG
        tilt_angle = 0.14391  # +/- one pixel shift for 3.45um Bayer pixels
    else:
        cameraType = DAHENG_MONO
        tilt_angle = 0.14391  # / 2  # +/- half pixel shift for 3.45um mono pixels
    h = cam.Height.get()
    w = cam.Width.get()

n_images = 4
n_images_sqrt = math.sqrt(n_images)
px_shifts = np.array([[-1, 1], [-1, -1], [1, -1], [1, 1]]) #
#currents = tilt_angle * px_shifts * np.array([1 / SAMx, 1 / SAMy])  # use with set_current (not used anymore)
angles = tilt_angle * px_shifts  # use with set_value in ICC-4C

tilt = 1  # pixel_shift
M = []
M0 = np.float32([[1, 0, 0], [0, 1, 0]])
M1 = np.float32([[1, 0, 0], [0, 1, tilt]])
M2 = np.float32([[1, 0, -tilt], [0, 1, tilt]])
M3 = np.float32([[1, 0, -tilt], [0, 1, 0]])
M.extend((M0, M1, M2, M3))

# Sub-pixel offsets per frame in native image pixels (from M matrices, ÷2)
_SUB_OFFSETS = [(0.0, 0.0), (0.0, 0.5), (-0.5, 0.5), (-0.5, 0.0)]

# Bayer channel masks — only needed for the colour Daheng path
if cameraType == DAHENG:
    masktile_R = np.array([[True, False], [False, False]])
    mask_R = np.tile(masktile_R, (h // 2, w // 2))
    masktile_G = np.array([[False, True], [True, False]])
    mask_G = np.tile(masktile_G, (h // 2, w // 2))
    masktile_B = np.array([[False, False], [False, True]])
    mask_B = np.tile(masktile_B, (h // 2, w // 2))

ROI_height = h // 8
ROI_width = w // 8

ROI_posx_min = ROI_width // 2 + w_line
ROI_posx_max = w - ROI_width // 2 - w_line
ROI_posy_min = ROI_height // 2 + w_line
ROI_posy_max = h - ROI_height // 2 - w_line

ROI_center_x = w // 2
ROI_center_y = h // 2

if sys.platform == 'win32':
    user32 = ctypes.windll.user32
    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
else:
    from PyQt5.QtWidgets import QApplication
    screen = QApplication.primaryScreen() if QApplication.instance() else None
    if screen:
        screensize = screen.size().width(), screen.size().height()
    else:
        screensize = (1920, 1080)
size_h_max = int(screensize[1]*0.4)  # makes sure that pictures fit on screen
resize_factor = h / size_h_max


class Runnable_Full_Cam(QRunnable):
    # Thread class controlling XPR + algo HR image
    def __init__(self):
        super().__init__()

    def run(self):
        global grab_mode
        global FPS, FPS2
        global frame_number, XPR_on

        global ROI_width, ROI_height, ROI_center_x, ROI_center_y
        global frame_CamImage, frame_CamZoom, frame_CamXPR, frame_CamXPR_numpy
        global channel, interpolation_mode, color_comparison
        global save_enabled, save_dir, save_set_ts
        global snr_enabled, gt_crop, snr_xpr_psnr, snr_xpr_ssim, snr_sub_psnr, snr_sub_ssim, snr_avg_psnr, snr_avg_ssim, _reg_match_img
        global snr_sub_ecc, snr_avg_ecc, snr_xpr_ecc
        global _sift_full_frame

        grab_mode = True
        if cameraType == BASLER:
            cam.StartGrabbing(1)

        _snr_sub_gray: list = [None] * n_images  # grayscale float32 per sub-frame for SNR (DAHENG color)

        while grab_mode:
            time_start_FPS = time.time()

            if XPR_on:
                icc4c.set_value([si_0.value, si_1.value],
                                [float(angles[frame_number, 0]), float(angles[frame_number, 1])])
            else:
                icc4c.set_value([si_0.value, si_1.value], [0, 0])

            # only software triggering used here:
            time.sleep(0.002) # , wait for XPR to reach position

            if cameraType == BASLER:
                cam.TriggerSoftware.Execute()
                grab = cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
                raw_image = grab.GetArray()
                frame = cv2.cvtColor(raw_image,
                                     cv2.COLOR_GRAY2RGB)  # makes manipulation for monochrome and color the same
            elif cameraType == DAHENG:
                cam.TriggerSoftware.send_command()
                raw_image = cam.data_stream[0].get_image()
                if raw_image is None:
                    continue
                rgb_image = raw_image.convert("RGB", convert_type=0)  # Bayer demosaic
                frame = rgb_image.get_numpy_array()
            else:  # DAHENG_MONO
                cam.TriggerSoftware.send_command()
                raw_image = cam.data_stream[0].get_image()
                if raw_image is None:
                    continue
                frame = cv2.cvtColor(raw_image.get_numpy_array(), cv2.COLOR_GRAY2RGB)

            bytesPerLine = 3 * w

            # Capture full frame for SIFT registration (once, at frame 0, when needed)
            if snr_enabled and gt_crop is not None and _reg_data is None and XPR_on and frame_number == 0:
                _sift_full_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

            px1 = int(ROI_center_x - ROI_width // 2)
            px2 = int(ROI_center_x + ROI_width // 2)
            py1 = int(ROI_center_y - ROI_height // 2)
            py2 = int(ROI_center_y + ROI_height // 2)
            frameZoom = frame[py1:py2, px1:px2].copy()

            if save_enabled and XPR_on:
                if frame_number == 0:
                    save_set_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(str(_save_filename(f"raw{frame_number}")),
                            cv2.cvtColor(frameZoom, cv2.COLOR_RGB2BGR))

            # add red frame around ROI
            frame[py1 - w_line:py1, px1 - w_line:px2 + w_line, :] = [255, 0, 0]
            frame[py2:py2 + w_line, px1 - w_line:px2 + w_line, :] = [255, 0, 0]
            frame[py1 - w_line:py2 + w_line, px1 - w_line:px1, :] = [255, 0, 0]
            frame[py1 - w_line:py2 + w_line, px2:px2 + w_line, :] = [255, 0, 0]

            mutex.lock()
            frame_CamImage = QImage(frame.data, w, h, bytesPerLine, QImage.Format.Format_RGB888)
            win.updatedCamImage.emit()

            frameZoom_resized = frameZoom.copy()

            if channel != RGB:
                if channel == R:
                    frameZoom_resized[:, :, 1:3] = 0
                elif channel == G:
                    frameZoom_resized[:, :, 0::2] = 0
                elif channel == B:
                    frameZoom_resized[:, :, 0:2] = 0
            mutex.unlock()

            size = win.CamZoom.size()

            if frame_number == 0:
                # cam_zoom image only shows frame 0, to avoid any image shifts

                mutex.lock()
                if np.array_equal(color_comparison, WHITE) or cameraType in (BASLER, DAHENG_MONO):
                    frameZoom_resized_send = frameZoom_resized.copy()

                elif np.array_equal(color_comparison, REDBLUE):
                    # force max possible value to 255 (uint8)
                    frame_tmp = np.maximum(
                        (-((np.sqrt(np.sum(np.power(frameZoom_resized[:, :, ::2].copy() - RED[::2], 2),
                                           axis=2)) * max_val) - 255)).astype(np.uint8), (
                            -((np.sqrt(np.sum(np.power(frameZoom_resized[:, :, ::2].copy() - BLUE[::2], 2),
                                              axis=2)) * max_val) - 255)).astype(np.uint8))

                    frame_tmp[frame_tmp < min_color_similarity] = 0
                    frame_tmp[frame_tmp >= min_color_similarity] = 1
                    frameZoom_resized_send = np.multiply(frameZoom_resized, frame_tmp[:, :, np.newaxis])

                else:
                    frame_tmp = (
                        -((np.sqrt(np.sum(np.power(frameZoom_resized[:, :, ::2].copy() - color_comparison[::2], 2),
                                          axis=2)) * max_val) - 255)).astype(np.uint8)
                    frame_tmp[frame_tmp < min_color_similarity] = 0
                    frame_tmp[frame_tmp >= min_color_similarity] = 1
                    frameZoom_resized_send = np.multiply(frameZoom_resized, frame_tmp[:, :, np.newaxis])

                if interpolation_mode == INTERPOLATED:
                    frameZoom_resized_send = cv2.resize(frameZoom_resized_send, (
                        2 * frameZoom_resized_send.shape[1], 2 * frameZoom_resized_send.shape[0]),
                                                        interpolation=cv2.INTER_LINEAR)

                bytesPerLine2 = 3 * frameZoom_resized_send.shape[1]
                frame_CamZoom = QImage.scaled(
                    QImage(frameZoom_resized_send.data, frameZoom_resized_send.shape[1],
                           frameZoom_resized_send.shape[0],
                           bytesPerLine2,
                           QImage.Format.Format_RGB888), size, transformMode=Qt.FastTransformation)
                win.updatedCamZoom.emit()
                win.updatedCamXPR.emit()
                mutex.unlock()

                mutex.lock()
                if XPR_on:
                    FPS2 = FPS / n_images
                else:
                    FPS2 = FPS
                win.updatedFPS2.emit()
                mutex.unlock()

            if XPR_on:
                # Algorithm creating HR image according color or mono camera
                if cameraType == DAHENG_MONO:
                    frame_raw = raw_image.get_numpy_array()  # H×W grayscale
                    height, width = frame_raw[py1: py2, px1: px2].shape

                    mutex.lock()
                    if frame_CamXPR_numpy.shape[0] != 2 * height or frame_CamXPR_numpy.shape[
                            1] != 2 * width or frame_number == 0:
                        frame_CamXPR_numpy = np.zeros((2 * height, 2 * width, n_images), dtype=np.uint8)
                    mutex.unlock()

                    # Store raw grayscale before warpAffine for SNR
                    if snr_enabled and gt_crop is not None and XPR_on:
                        _snr_sub_gray[frame_number] = cv2.cvtColor(
                            frameZoom, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

                    frame_CamXPR_numpy[::2, ::2, frame_number] = frame_raw[py1: py2, px1: px2]
                    frame_CamXPR_numpy[:, :, frame_number] = cv2.warpAffine(
                        frame_CamXPR_numpy[:, :, frame_number], M[frame_number],
                        (2 * width, 2 * height), borderMode=cv2.BORDER_REFLECT_101)

                    if frame_number == n_images - 1:
                        frame_CamXPR_numpy_HR = np.sum(frame_CamXPR_numpy, axis=2, dtype=np.uint8)
                        if snr_enabled and gt_crop is not None:
                            # Per-sub-image SNR from stored raw grayscale with beam-shift offsets
                            for _i in range(n_images):
                                if _snr_sub_gray[_i] is not None:
                                    snr_sub_psnr[_i], snr_sub_ssim[_i], snr_sub_ecc[_i] = _compute_snr_pair(
                                        gt_crop, _snr_sub_gray[_i], pixel_offset=_SUB_OFFSETS[_i])
                            # Average of sub-images at native resolution
                            if (all(f is not None for f in _snr_sub_gray) and
                                    len(set(f.shape for f in _snr_sub_gray)) == 1):
                                avg_raw = np.mean(np.stack(_snr_sub_gray, axis=2), axis=2)
                            else:
                                avg_raw = cv2.cvtColor(frameZoom, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                            snr_avg_psnr, snr_avg_ssim, snr_avg_ecc = _compute_snr_pair(gt_crop, avg_raw, store_match=True)
                            # XPR result at 2× resolution
                            xpr_gray = frame_CamXPR_numpy_HR.astype(np.float32) / 255.0
                            snr_xpr_psnr, snr_xpr_ssim, snr_xpr_ecc = _compute_snr_pair(gt_crop, xpr_gray)
                            win.updatedSNR.emit()
                        if save_enabled:
                            cv2.imwrite(str(_save_filename("xpr")), frame_CamXPR_numpy_HR)
                        frame_CamXPR = QImage.scaled(
                            QImage(frame_CamXPR_numpy_HR.data, frame_CamXPR_numpy_HR.shape[1],
                                   frame_CamXPR_numpy_HR.shape[0],
                                   frame_CamXPR_numpy_HR.shape[1], QImage.Format.Format_Grayscale8),
                            size, transformMode=Qt.FastTransformation)
                        win.updatedCamXPR.emit()

                    mutex.lock()
                    if XPR_on:
                        frame_number = (frame_number + 1) % n_images
                    mutex.unlock()

                elif cameraType == DAHENG:
                    frame_raw = raw_image.get_numpy_array()
                    height, width = frame_raw[py1: py2, px1: px2].shape

                    mutex.lock()
                    # enforce correct
                    if frame_CamXPR_numpy.shape[0] != frameZoom_resized.shape[0] or frame_CamXPR_numpy.shape[1] != \
                            frameZoom_resized.shape[1] or frame_number == 0:
                        frame_CamXPR_numpy = np.zeros((frameZoom_resized.shape[0], frameZoom_resized.shape[1], 3),
                                                      dtype=np.uint8)
                    mutex.unlock()

                    # Store this frame's grayscale for per-sub-image SNR
                    if snr_enabled and gt_crop is not None and XPR_on:
                        _snr_sub_gray[frame_number] = cv2.cvtColor(
                            frameZoom, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

                    R_channel = np.where(mask_R, np.copy(frame_raw), 0)
                    G_channel = np.where(mask_G, np.copy(frame_raw), 0)
                    B_channel = np.where(mask_B, np.copy(frame_raw), 0)

                    if channel == RGB or channel == R:
                        frame_translated_R = cv2.warpAffine(R_channel[py1: py2, px1: px2], M[frame_number],
                                                            (width, height),
                                                            borderMode=cv2.BORDER_REFLECT_101)
                        frame_CamXPR_numpy[
                        frameZoom_resized.shape[0] // 2 - frameZoom.shape[0] // 2:frameZoom_resized.shape[0] // 2 -
                                                                                  frameZoom.shape[0] // 2 +
                                                                                  frameZoom.shape[0],
                        frameZoom_resized.shape[1] // 2 - frameZoom.shape[1] // 2:frameZoom_resized.shape[1] // 2 -
                                                                                  frameZoom.shape[1] // 2 +
                                                                                  frameZoom.shape[1],
                        0] += frame_translated_R

                    if channel == RGB or channel == G:
                        frame_translated_G = cv2.warpAffine(G_channel[py1: py2, px1: px2], M[frame_number],
                                                            (width, height),
                                                            borderMode=cv2.BORDER_REFLECT_101)
                        frame_CamXPR_numpy[
                        frameZoom_resized.shape[0] // 2 - frameZoom.shape[0] // 2:frameZoom_resized.shape[0] // 2 -
                                                                                  frameZoom.shape[0] // 2 +
                                                                                  frameZoom.shape[0],
                        frameZoom_resized.shape[1] // 2 - frameZoom.shape[1] // 2:frameZoom_resized.shape[1] // 2 -
                                                                                  frameZoom.shape[1] // 2 +
                                                                                  frameZoom.shape[1],
                        1] += frame_translated_G // 2

                    if channel == RGB or channel == B:
                        frame_translated_B = cv2.warpAffine(B_channel[py1: py2, px1: px2], M[frame_number],
                                                            (width, height),
                                                            borderMode=cv2.BORDER_REFLECT_101)
                        frame_CamXPR_numpy[
                        frameZoom_resized.shape[0] // 2 - frameZoom.shape[0] // 2:frameZoom_resized.shape[0] // 2 -
                                                                                  frameZoom.shape[0] // 2 +
                                                                                  frameZoom.shape[0],
                        frameZoom_resized.shape[1] // 2 - frameZoom.shape[1] // 2:frameZoom_resized.shape[1] // 2 -
                                                                                  frameZoom.shape[1] // 2 +
                                                                                  frameZoom.shape[1],
                        2] += frame_translated_B

                    frame_CamXPR_numpy[:, [0, -1], :] = frame_CamXPR_numpy[[0, -1], :] = 0

                    bytesPerLine2 = 3 * frame_CamXPR_numpy.shape[1]
                    if frame_number == n_images - 1:

                        if np.array_equal(color_comparison, WHITE):
                            frame_CamXPR_numpy_send = frame_CamXPR_numpy.copy()

                        elif np.array_equal(color_comparison, REDBLUE):
                            frame_tmp = np.maximum(
                                (-((np.sqrt(np.sum(np.power(frame_CamXPR_numpy[:, :, ::2].copy() - RED[::2], 2),
                                                   axis=2)) * max_val) - 255)).astype(np.uint8), (
                                    -((np.sqrt(np.sum(np.power(frame_CamXPR_numpy[:, :, ::2].copy() - BLUE[::2], 2),
                                                      axis=2)) * max_val) - 255)).astype(np.uint8))

                            frame_tmp[frame_tmp < min_color_similarity] = 0
                            frame_tmp[frame_tmp >= min_color_similarity] = 1
                            frame_CamXPR_numpy_send = np.multiply(frame_CamXPR_numpy, frame_tmp[:, :, np.newaxis])

                        else:
                            frame_tmp = (-((np.sqrt( np.sum(np.power(frame_CamXPR_numpy[:, :,::2].copy() - color_comparison[::2], 2), axis=2)) * max_val) - 255)).astype(np.uint8)

                            frame_tmp[frame_tmp < min_color_similarity] = 0
                            frame_tmp[frame_tmp >= min_color_similarity] = 1
                            frame_CamXPR_numpy_send = np.multiply(frameZoom_resized, frame_tmp[:, :, np.newaxis])

                        if snr_enabled and gt_crop is not None:
                            # Per-sub-image SNR from stored grayscale frames with beam-shift offsets
                            for _i in range(n_images):
                                if _snr_sub_gray[_i] is not None:
                                    snr_sub_psnr[_i], snr_sub_ssim[_i], snr_sub_ecc[_i] = _compute_snr_pair(
                                        gt_crop, _snr_sub_gray[_i], pixel_offset=_SUB_OFFSETS[_i])
                            # Average of sub-images
                            if (all(f is not None for f in _snr_sub_gray) and
                                    len(set(f.shape for f in _snr_sub_gray)) == 1):
                                avg_raw = np.mean(np.stack(_snr_sub_gray, axis=2), axis=2)
                            else:
                                avg_raw = cv2.cvtColor(frameZoom, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                            snr_avg_psnr, snr_avg_ssim, snr_avg_ecc = _compute_snr_pair(gt_crop, avg_raw, store_match=True)
                            xpr_gray = cv2.cvtColor(frame_CamXPR_numpy, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                            snr_xpr_psnr, snr_xpr_ssim, snr_xpr_ecc = _compute_snr_pair(gt_crop, xpr_gray)
                            win.updatedSNR.emit()
                        if save_enabled:
                            cv2.imwrite(str(_save_filename("xpr")),
                                        cv2.cvtColor(frame_CamXPR_numpy_send, cv2.COLOR_RGB2BGR))

                        frame_CamXPR = QImage.scaled(
                            QImage(frame_CamXPR_numpy_send.data, frame_CamXPR_numpy.shape[1],
                                   frame_CamXPR_numpy.shape[0],
                                   bytesPerLine2, QImage.Format.Format_RGB888), size,
                            transformMode=Qt.FastTransformation)
                        win.updatedCamXPR.emit()

                    mutex.lock()
                    if XPR_on:
                        frame_number = (frame_number + 1) % n_images
                    mutex.unlock()

                elif cameraType == BASLER:
                    frame_raw = np.copy(raw_image)
                    height, width = frame_raw[py1: py2, px1: px2].shape

                    mutex.lock()
                    # create higher resolution frame
                    if frame_CamXPR_numpy.shape[0] != 2 * height or frame_CamXPR_numpy.shape[
                        1] != 2 * width or frame_number == 0:
                        frame_CamXPR_numpy = np.zeros((2 * height, 2 * width, n_images), dtype=np.uint8)
                    mutex.unlock()

                    # Store raw grayscale before warpAffine for SNR
                    if snr_enabled and gt_crop is not None and XPR_on:
                        _snr_sub_gray[frame_number] = cv2.cvtColor(
                            frameZoom, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

                    frame_CamXPR_numpy[::2, ::2, frame_number] = frame_raw[py1: py2, px1: px2]
                    frame_CamXPR_numpy[:, :, frame_number] = cv2.warpAffine(frame_CamXPR_numpy[:, :, frame_number],
                                                                            M[frame_number], (2 * width, 2 * height),
                                                                            borderMode=cv2.BORDER_REFLECT_101)

                    if frame_number == n_images - 1:
                        frame_CamXPR_numpy_HR = np.sum(frame_CamXPR_numpy, axis=2, dtype=np.uint8)
                        if snr_enabled and gt_crop is not None:
                            # Per-sub-image SNR from stored raw grayscale with beam-shift offsets
                            for _i in range(n_images):
                                if _snr_sub_gray[_i] is not None:
                                    snr_sub_psnr[_i], snr_sub_ssim[_i], snr_sub_ecc[_i] = _compute_snr_pair(
                                        gt_crop, _snr_sub_gray[_i], pixel_offset=_SUB_OFFSETS[_i])
                            # Average of sub-images at native resolution
                            if (all(f is not None for f in _snr_sub_gray) and
                                    len(set(f.shape for f in _snr_sub_gray)) == 1):
                                avg_raw = np.mean(np.stack(_snr_sub_gray, axis=2), axis=2)
                            else:
                                avg_raw = cv2.cvtColor(frameZoom, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                            snr_avg_psnr, snr_avg_ssim, snr_avg_ecc = _compute_snr_pair(gt_crop, avg_raw, store_match=True)
                            # XPR result at 2× resolution
                            xpr_gray = frame_CamXPR_numpy_HR.astype(np.float32) / 255.0
                            snr_xpr_psnr, snr_xpr_ssim, snr_xpr_ecc = _compute_snr_pair(gt_crop, xpr_gray)
                            win.updatedSNR.emit()
                        if save_enabled:
                            cv2.imwrite(str(_save_filename("xpr")), frame_CamXPR_numpy_HR)
                        frame_CamXPR = QImage.scaled(
                            QImage(frame_CamXPR_numpy_HR.data, frame_CamXPR_numpy_HR.shape[1],
                                   frame_CamXPR_numpy_HR.shape[0],
                                   frame_CamXPR_numpy_HR.shape[1], QImage.Format.Format_Grayscale8), size,
                            transformMode=Qt.FastTransformation)
                        win.updatedCamXPR.emit()

                    mutex.lock()
                    if XPR_on:
                        frame_number = (frame_number + 1) % n_images
                    mutex.unlock()
            else:
                # If XPR is deactivated: shows the same pictures
                frame_CamXPR = frame_CamZoom
                win.updatedCamXPR.emit()

            time_end_FPS = time.time()
            FPS_current = round(1 / (time_end_FPS - time_start_FPS), 1)

            mutex.lock()
            FPS = (1 - FPS_n) * FPS + FPS_n * FPS_current  # filtered value for fps
            win.updatedFPS.emit()
            mutex.unlock()

        mutex.lock()
        FPS = 0
        FPS2 = 0
        mutex.unlock()
        win.updatedFPS.emit()
        win.updatedFPS2.emit()

        icc4c.set_value([si_0.value, si_1.value], [0, 0])

        if cameraType == BASLER:
            cam.StopGrabbing()


class GTPickerDialog(QDialog):
    """Modal dialog for selecting a rectangular region on a ground truth image."""

    def __init__(self, gt_image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Ground Truth Region")
        self._gt = gt_image          # original float32 [0,1] grayscale
        self._crop = None            # will be set on confirm

        orig_h, orig_w = gt_image.shape
        max_display_h = 700
        max_display_w = 900
        self._scale = min(max_display_h / orig_h, max_display_w / orig_w, 1.0)
        disp_h = int(orig_h * self._scale)
        disp_w = int(orig_w * self._scale)

        # Convert to displayable QPixmap (scale + convert to uint8 RGB)
        disp = cv2.resize((gt_image * 255).astype(np.uint8), (disp_w, disp_h),
                          interpolation=cv2.INTER_AREA)
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
        qimg = QImage(disp_rgb.data, disp_w, disp_h, 3 * disp_w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)

        self._label = QLabel()
        self._label.setPixmap(self._pixmap)
        self._label.setFixedSize(disp_w, disp_h)
        self._label.setCursor(Qt.CrossCursor)
        self._label.mousePressEvent = self._on_press
        self._label.mouseMoveEvent = self._on_move
        self._label.mouseReleaseEvent = self._on_release

        self._rubber = QRubberBand(QRubberBand.Rectangle, self._label)
        self._origin = None
        self._sel = None   # (x1, y1, x2, y2) in original image coords

        self._info = QLabel("Click and drag to select a region")
        self._confirm = QPushButton("Confirm Selection")
        self._confirm.setEnabled(False)
        self._confirm.clicked.connect(self._on_confirm)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._confirm)
        btn_row.addWidget(cancel)

        layout = QVBoxLayout()
        layout.addWidget(self._label)
        layout.addWidget(self._info)
        layout.addLayout(btn_row)
        self.setLayout(layout)

    def _on_press(self, event):
        self._origin = event.pos()
        self._rubber.setGeometry(QRect(self._origin, QSize()))
        self._rubber.show()

    def _on_move(self, event):
        if self._origin is not None:
            self._rubber.setGeometry(QRect(self._origin, event.pos()).normalized())

    def _on_release(self, event):
        if self._origin is None:
            return
        rect = QRect(self._origin, event.pos()).normalized()
        self._rubber.setGeometry(rect)

        # Convert display coords to original image coords
        s = self._scale
        x1 = max(0, int(rect.x() / s))
        y1 = max(0, int(rect.y() / s))
        x2 = min(self._gt.shape[1], int((rect.x() + rect.width()) / s))
        y2 = min(self._gt.shape[0], int((rect.y() + rect.height()) / s))

        if (x2 - x1) > 2 and (y2 - y1) > 2:
            self._sel = (x1, y1, x2, y2)
            self._info.setText(f"Selection: x={x1} y={y1}  w={x2-x1} h={y2-y1} px")
            self._confirm.setEnabled(True)
        else:
            self._sel = None
            self._info.setText("Selection too small — drag a larger area")
            self._confirm.setEnabled(False)

        self._origin = None

    def _on_confirm(self):
        if self._sel is not None:
            x1, y1, x2, y2 = self._sel
            self._crop = self._gt[y1:y2, x1:x2].copy()
            self.accept()

    def get_crop(self) -> np.ndarray:
        return self._crop


def _lin_norm(ref: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Find a,b minimising ||ref - (a*img + b)||², apply to img, clip to [0,1]."""
    img64 = img.ravel().astype(np.float64)
    ref64 = ref.ravel().astype(np.float64)
    var_img = float(np.var(img64))
    if var_img < 1e-10:
        return np.clip(np.full_like(img, float(ref64.mean())), 0, 1).astype(np.float32)
    a = float(np.cov(img64, ref64)[0, 1] / var_img)
    b = float(ref64.mean() - a * img64.mean())
    return np.clip(a * img + b, 0, 1).astype(np.float32)


def _best_orient(gt: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Try all 8 orientations (4 rotations × flip) of gt resized to img shape.
    Returns the orientation that minimises MSE after linear normalisation.
    Handles cameras/prints that are rotated or mirrored relative to the GT."""
    best_mse = float('inf')
    best_gt = None
    h, w = img.shape
    for k in range(4):
        for flip in (False, True):
            cand = np.rot90(gt, k)
            if flip:
                cand = np.fliplr(cand)
            cand = cv2.resize(cand, (w, h), interpolation=cv2.INTER_AREA)
            img_fit = _lin_norm(cand, img)
            mse = float(np.mean((cand - img_fit) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_gt = cand
    return best_gt


def _psf_blur(tmpl: np.ndarray) -> np.ndarray:
    """Apply small Gaussian blur to GT template to approximate camera PSF."""
    k = max(3, int(min(tmpl.shape[1], tmpl.shape[0]) * 0.02) | 1)
    return cv2.GaussianBlur(tmpl, (k, k), 0)


def _multiscale_template_match(gt_crop: np.ndarray, full_frame: np.ndarray,
                                roi_rect: tuple):
    """Multi-scale template matching on the FULL camera frame.

    The GT crop may be the entire printed chart (including the cross), not just
    the barcode group visible in the ROI.  We therefore search over a wide
    scale range and require the ROI center to fall INSIDE the matched region
    (the chart must cover the ROI).

    Phase 1:  Coarse multi-scale position search (2 aspect ratios × N scales).
    Phase 1b: Orientation at the found position + scale (all 8, fair NCC).
    Phase 2:  Fine scale + position refinement at full resolution.

    Returns (gt_oriented, tx, ty, tw, th, ncc, label, debug_str) or None.
    tx, ty = top-left of matched template in full-frame coordinates.
    tw, th = template dimensions at the matched scale."""
    px1, py1, px2, py2 = roi_rect
    roi_w, roi_h = px2 - px1, py2 - py1
    roi_cx, roi_cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
    fh, fw = full_frame.shape[:2]
    gt_h, gt_w = gt_crop.shape[:2]

    # --- Phase 1: Coarse multi-scale position search on the full frame ---
    CF = 4
    frame_small = cv2.resize(full_frame, (fw // CF, fh // CF),
                             interpolation=cv2.INTER_AREA)
    fsh, fsw = frame_small.shape[:2]
    roi_cx_c, roi_cy_c = roi_cx / CF, roi_cy / CF

    # Scale range: from GT covering ~10% of the frame to ~90%
    max_frame_scale = min(fw / gt_w, fh / gt_h)
    scale_lo = 0.05 * max_frame_scale
    scale_hi = 0.85 * max_frame_scale
    N_SCALES = 25

    best_p1_score = -1.0
    best_p1 = None
    p1_debug_lines = []

    for k in (0, 1):  # two distinct aspect ratios
        cand = np.rot90(gt_crop, k)
        ch, cw = cand.shape[:2]

        for scale in np.linspace(scale_lo, scale_hi, N_SCALES):
            tw_c = max(1, int(cw * scale / CF))
            th_c = max(1, int(ch * scale / CF))
            if tw_c >= fsw - 2 or th_c >= fsh - 2 or tw_c < 16 or th_c < 16:
                continue

            tmpl = _psf_blur(
                cv2.resize(cand, (tw_c, th_c),
                           interpolation=cv2.INTER_AREA).astype(np.float32))
            res = cv2.matchTemplate(frame_small, tmpl, cv2.TM_CCOEFF_NORMED)
            _, mx_val, _, mx_loc = cv2.minMaxLoc(res)

            # ROI containment: the ROI center must fall inside the matched region
            tx_c, ty_c = mx_loc
            if not (tx_c <= roi_cx_c <= tx_c + tw_c and
                    ty_c <= roi_cy_c <= ty_c + th_c):
                continue

            if mx_val > best_p1_score:
                best_p1_score = mx_val
                best_p1 = (k, scale, mx_loc, tw_c, th_c, mx_val)

    if best_p1 is None:
        return None

    p1_k, p1_scale, p1_loc, p1_tw, p1_th, p1_ncc = best_p1
    # Full-frame coordinates of match center
    cx_full = (p1_loc[0] + p1_tw / 2.0) * CF
    cy_full = (p1_loc[1] + p1_th / 2.0) * CF

    # --- Phase 1b: Orientation search at the found position + scale ---
    orient_scores = []
    best_orient_score = -1.0
    best_orient = None
    ORIENT_PRIOR_DECAY = 0.95  # 5% penalty per rotation step

    for k in range(4):
        for flip in (False, True):
            cand = np.rot90(gt_crop, k)
            if flip:
                cand = np.fliplr(cand)
            ch, cw = cand.shape[:2]
            lbl = f"rot{k}" + ("/flip" if flip else "")

            tw_f = max(1, int(cw * p1_scale))
            th_f = max(1, int(ch * p1_scale))
            if tw_f < 16 or th_f < 16:
                orient_scores.append(f"{lbl}=skip")
                continue

            # Local search region around Phase 1 center (100% padding)
            half_w, half_h = tw_f, th_f
            lx1 = max(0, int(cx_full - half_w))
            ly1 = max(0, int(cy_full - half_h))
            lx2 = min(fw, int(cx_full + half_w))
            ly2 = min(fh, int(cy_full + half_h))
            if lx2 - lx1 < tw_f or ly2 - ly1 < th_f:
                orient_scores.append(f"{lbl}=skip")
                continue

            local = full_frame[ly1:ly2, lx1:lx2]
            tmpl = _psf_blur(
                cv2.resize(cand, (tw_f, th_f),
                           interpolation=cv2.INTER_AREA).astype(np.float32))
            res = cv2.matchTemplate(local, tmpl, cv2.TM_CCOEFF_NORMED)
            _, mx_val, _, _ = cv2.minMaxLoc(res)

            steps = min(k, 4 - k)
            if flip:
                steps += 1
            prior = ORIENT_PRIOR_DECAY ** steps
            scored = mx_val * prior
            orient_scores.append(f"{lbl}={mx_val:.4f}*{prior:.2f}={scored:.4f}")

            if scored > best_orient_score:
                best_orient_score = scored
                best_orient = (cand, lbl, p1_scale, tw_f, th_f)

    if best_orient is None:
        debug = (f"TmplMatch: no orient match\n"
                 f"  orient: {', '.join(orient_scores)}")
        return None

    cand, label, base_scale, tw_f, th_f = best_orient
    ch, cw = cand.shape[:2]

    # --- Phase 2: Fine scale + position refinement at full resolution ---
    fine_pad = int(max(tw_f, th_f) * 0.3)
    cx_tl = cx_full - tw_f / 2.0
    cy_tl = cy_full - th_f / 2.0

    best_fine_ncc = -1.0
    best_fine = None

    for scale in np.linspace(base_scale * 0.85, base_scale * 1.15, 15):
        tw = max(1, int(cw * scale))
        th = max(1, int(ch * scale))
        if tw < 16 or th < 16:
            continue
        fx1 = max(0, int(cx_tl - fine_pad))
        fy1 = max(0, int(cy_tl - fine_pad))
        fx2 = min(fw, int(cx_tl + tw + fine_pad))
        fy2 = min(fh, int(cy_tl + th + fine_pad))
        if fx2 - fx1 < tw or fy2 - fy1 < th:
            continue
        region = full_frame[fy1:fy2, fx1:fx2]
        tmpl = _psf_blur(
            cv2.resize(cand, (tw, th),
                       interpolation=cv2.INTER_AREA).astype(np.float32))
        res = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
        _, mx_val, _, mx_loc = cv2.minMaxLoc(res)
        if mx_val > best_fine_ncc:
            best_fine_ncc = mx_val
            best_fine = (fx1 + mx_loc[0], fy1 + mx_loc[1],
                         tw, th, mx_val, scale)

    if best_fine is None:
        tw = max(1, int(cw * base_scale))
        th = max(1, int(ch * base_scale))
        best_fine = (int(cx_tl), int(cy_tl), tw, th, 0.0, base_scale)

    tx, ty, tw, th, fine_ncc, fine_scale = best_fine

    debug = (f"TmplMatch: frame {fw}x{fh}, gt {gt_w}x{gt_h}\n"
             f"  scale range [{scale_lo:.4f}, {scale_hi:.4f}], "
             f"N={N_SCALES}, CF={CF}\n"
             f"  Phase 1: k={p1_k}, scale={p1_scale:.4f}, "
             f"NCC={p1_ncc:.4f}, pos=({cx_full:.0f},{cy_full:.0f})\n"
             f"  Phase 1b orient: {', '.join(orient_scores)}\n"
             f"  best orient: {label}\n"
             f"  Phase 2: scale={fine_scale:.4f}, tmpl {tw}x{th}, "
             f"pos ({tx},{ty}), NCC={fine_ncc:.4f}")
    return cand, tx, ty, tw, th, fine_ncc, label, debug


def _build_match_visualization(full_frame: np.ndarray, tx: int, ty: int,
                                tw: int, th: int, roi_rect: tuple,
                                label: str, ncc: float) -> np.ndarray:
    """Draw match rectangle (green) and ROI (red) on the full frame. Returns uint8 RGB."""
    fh, fw = full_frame.shape[:2]
    max_dim = 800
    s = min(max_dim / fw, max_dim / fh, 1.0)
    if s < 1.0:
        vis = cv2.resize((full_frame * 255).astype(np.uint8),
                         (int(fw * s), int(fh * s)), interpolation=cv2.INTER_AREA)
    else:
        vis = (full_frame * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    # Match rectangle (green)
    cv2.rectangle(vis, (int(tx * s), int(ty * s)),
                  (int((tx + tw) * s), int((ty + th) * s)), (0, 255, 0), 2)
    # ROI rectangle (red)
    px1, py1, px2, py2 = roi_rect
    cv2.rectangle(vis, (int(px1 * s), int(py1 * s)),
                  (int(px2 * s), int(py2 * s)), (255, 0, 0), 2)
    cv2.putText(vis, f"{label} NCC={ncc:.3f}",
                (int(tx * s), max(int(ty * s) - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis


def _register_gt(gt_crop: np.ndarray, img: np.ndarray,
                 pixel_offset=(0.0, 0.0), store_match=False):
    """Register gt_crop onto img via multi-scale template matching (full frame) + ECC (ROI).
    Initial call: template match on full frame, ECC MOTION_TRANSLATION refinement.
    Subsequent calls: apply cached homography + beam offset.
    pixel_offset: sub-pixel (dx, dy) for beam-shifted sub-images.
    store_match: if True, re-run ECC refinement and update cache; save aligned GT.
    Returns (gt_warped, (y1,y2,x1,x2), ncc_score) on success, or None."""
    global _reg_data, _reg_status, _reg_debug, _sift_full_frame, _reg_match_img

    h, w = img.shape[:2]
    img_f32 = img.astype(np.float32) if img.dtype != np.float32 else img

    # Snapshot to avoid race with invalidate_registration() on GUI thread
    cached = _reg_data

    # --- Per-frame: apply cached homography + beam offset ---
    if cached is not None:
        gt_rot = cached["gt_rot"]
        src_h, src_w = cached["src_h"], cached["src_w"]
        H = cached["H_roi"].copy()
        ncc_score = cached["ncc"]
        # Scale homography for current resolution (handles XPR 2× case)
        if w != src_w or h != src_h:
            sx, sy = w / src_w, h / src_h
            S_out = np.diag([sx, sy, 1.0])
            S_in = np.diag([1.0 / sx, 1.0 / sy, 1.0])
            H = S_out @ H @ S_in
        # Apply beam offset
        T_offset = np.eye(3, dtype=np.float64)
        T_offset[0, 2] = pixel_offset[0]
        T_offset[1, 2] = pixel_offset[1]
        H = T_offset @ H
        # Resize GT to current img dims and warp
        gt_resized = cv2.resize(gt_rot, (w, h),
                                interpolation=cv2.INTER_AREA).astype(np.float32)
        # If store_match (average image), try ECC refinement
        if store_match:
            gt_warped_pre = cv2.warpPerspective(
                gt_resized, H, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)
            # Compute valid region for NCC/ECC (GT may not fill entire ROI)
            mask_pre = cv2.warpPerspective(
                np.ones((h, w), dtype=np.float32), H, (w, h)) > 0.5
            rows_vp = np.any(mask_pre, axis=1)
            cols_vp = np.any(mask_pre, axis=0)
            if rows_vp.any() and cols_vp.any():
                vy1p = int(np.where(rows_vp)[0][0]);  vy2p = int(np.where(rows_vp)[0][-1]) + 1
                vx1p = int(np.where(cols_vp)[0][0]);  vx2p = int(np.where(cols_vp)[0][-1]) + 1
            else:
                vy1p, vy2p, vx1p, vx2p = 0, h, 0, w
            vhp, vwp = vy2p - vy1p, vx2p - vx1p
            if min(vhp, vwp) >= 8:
                ncc_before = float(cv2.matchTemplate(
                    img_f32[vy1p:vy2p, vx1p:vx2p],
                    gt_warped_pre[vy1p:vy2p, vx1p:vx2p],
                    cv2.TM_CCOEFF_NORMED)[0, 0])
            else:
                ncc_before = 0.0
            try:
                blur_k = max(3, int(min(vhp, vwp) * 0.02) | 1)
                img_blur = cv2.GaussianBlur(
                    img_f32[vy1p:vy2p, vx1p:vx2p], (blur_k, blur_k), 0)
                gt_blur = cv2.GaussianBlur(
                    gt_warped_pre[vy1p:vy2p, vx1p:vx2p], (blur_k, blur_k), 0)
                warp_ecc = np.eye(2, 3, dtype=np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
                _, warp_ecc = cv2.findTransformECC(
                    img_blur, gt_blur, warp_ecc, cv2.MOTION_TRANSLATION, criteria)
                gt_warped_ecc = cv2.warpAffine(
                    gt_warped_pre, warp_ecc, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)
                dtx_e, dty_e = float(warp_ecc[0, 2]), float(warp_ecc[1, 2])
                evy1 = max(0, int(round(vy1p + dty_e)));  evy2 = min(h, int(round(vy2p + dty_e)))
                evx1 = max(0, int(round(vx1p + dtx_e)));  evx2 = min(w, int(round(vx2p + dtx_e)))
                if min(evy2 - evy1, evx2 - evx1) >= 8:
                    ncc_after = float(cv2.matchTemplate(
                        img_f32[evy1:evy2, evx1:evx2],
                        gt_warped_ecc[evy1:evy2, evx1:evx2],
                        cv2.TM_CCOEFF_NORMED)[0, 0])
                else:
                    ncc_after = 0.0
                if ncc_after > ncc_before:
                    # ECC improved — compose refinement into cached H
                    T_refine = np.eye(3, dtype=np.float64)
                    T_refine[0, 2] = float(warp_ecc[0, 2])
                    T_refine[1, 2] = float(warp_ecc[1, 2])
                    # Update cache (at source resolution)
                    H_base = cached["H_roi"].copy()
                    if w != src_w or h != src_h:
                        sx, sy = w / src_w, h / src_h
                        T_refine_src = np.eye(3, dtype=np.float64)
                        T_refine_src[0, 2] = T_refine[0, 2] / sx
                        T_refine_src[1, 2] = T_refine[1, 2] / sy
                        cached["H_roi"] = T_refine_src @ H_base
                    else:
                        cached["H_roi"] = T_refine @ H_base
                    cached["ncc"] = ncc_after
                    ncc_score = ncc_after
                    H = T_refine @ H  # update current H too
            except cv2.error:
                pass
        gt_warped = cv2.warpPerspective(
            gt_resized, H, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)
        # Compute bbox from warp mask
        mask_warp = cv2.warpPerspective(
            np.ones((h, w), dtype=np.float32), H, (w, h)) > 0.5
        rows = np.any(mask_warp, axis=1)
        cols = np.any(mask_warp, axis=0)
        if not rows.any() or not cols.any():
            _reg_status = "Fallback"
            return None
        y1 = int(np.where(rows)[0][0]);  y2 = int(np.where(rows)[0][-1]) + 1
        x1 = int(np.where(cols)[0][0]);  x2 = int(np.where(cols)[0][-1]) + 1
        _reg_status = f"{cached['method']} (ncc={ncc_score:.3f})"
        if store_match:
            _reg_match_img = gt_warped[y1:y2, x1:x2]
        return gt_warped, (y1, y2, x1, x2), float(ncc_score)

    # --- Initial registration: multi-scale template matching on full frame ---
    if _sift_full_frame is None:
        _reg_status = "Fallback"
        _reg_debug = "No full frame available (waiting for frame 0)"
        return None

    full_frame = _sift_full_frame
    _sift_full_frame = None  # free memory

    px1 = int(ROI_center_x - ROI_width // 2)
    py1 = int(ROI_center_y - ROI_height // 2)
    px2 = px1 + int(ROI_width)
    py2 = py1 + int(ROI_height)

    result = _multiscale_template_match(gt_crop, full_frame, (px1, py1, px2, py2))
    if result is None:
        _reg_status = "Fallback"
        _reg_debug += "\nTmplMatch: no valid match across all orientations"
        return None

    gt_oriented, tx, ty, tw, th, ncc_match, label, tm_debug = result
    _reg_debug = tm_debug

    # Build H_roi: simple scale + translate (GT-resized → ROI coords)
    H_roi = np.array([
        [tw / w,  0.0,    tx - px1],
        [0.0,     th / h, ty - py1],
        [0.0,     0.0,    1.0     ]
    ], dtype=np.float64)

    # Warp GT and compute valid region (GT may not fill the entire ROI)
    gt_resized = cv2.resize(gt_oriented, (w, h),
                            interpolation=cv2.INTER_AREA).astype(np.float32)
    gt_warped = cv2.warpPerspective(
        gt_resized, H_roi, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)
    mask_warp = cv2.warpPerspective(
        np.ones((h, w), dtype=np.float32), H_roi, (w, h)) > 0.5
    rows_v = np.any(mask_warp, axis=1)
    cols_v = np.any(mask_warp, axis=0)
    if not rows_v.any() or not cols_v.any():
        _reg_status = "Fallback"
        _reg_debug += "\n>> Warp produced empty region"
        return None
    vy1 = int(np.where(rows_v)[0][0]);  vy2 = int(np.where(rows_v)[0][-1]) + 1
    vx1 = int(np.where(cols_v)[0][0]);  vx2 = int(np.where(cols_v)[0][-1]) + 1
    vh, vw = vy2 - vy1, vx2 - vx1

    # NCC on valid region only (avoid degenerate result from zero-padded areas)
    if min(vh, vw) >= 8:
        ncc_score = float(cv2.matchTemplate(
            img_f32[vy1:vy2, vx1:vx2], gt_warped[vy1:vy2, vx1:vx2],
            cv2.TM_CCOEFF_NORMED)[0, 0])
    else:
        ncc_score = 0.0
    _reg_debug += (f"\n>> H_roi NCC (pre-ECC, valid {vw}x{vh} of {w}x{h}): "
                   f"{ncc_score:.4f}")

    # ECC MOTION_TRANSLATION refinement on valid region only
    try:
        blur_k = max(3, int(min(vh, vw) * 0.02) | 1)
        img_blur = cv2.GaussianBlur(img_f32[vy1:vy2, vx1:vx2], (blur_k, blur_k), 0)
        gt_blur = cv2.GaussianBlur(gt_warped[vy1:vy2, vx1:vx2], (blur_k, blur_k), 0)
        warp_ecc = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
        _, warp_ecc = cv2.findTransformECC(
            img_blur, gt_blur, warp_ecc, cv2.MOTION_TRANSLATION, criteria)
        dtx, dty = float(warp_ecc[0, 2]), float(warp_ecc[1, 2])
        # Apply refinement to full-image warp
        gt_warped_ecc = cv2.warpAffine(
            gt_warped, warp_ecc, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)
        # NCC on shifted valid region
        evy1 = max(0, int(round(vy1 + dty)));  evy2 = min(h, int(round(vy2 + dty)))
        evx1 = max(0, int(round(vx1 + dtx)));  evx2 = min(w, int(round(vx2 + dtx)))
        if min(evy2 - evy1, evx2 - evx1) >= 8:
            ncc_after = float(cv2.matchTemplate(
                img_f32[evy1:evy2, evx1:evx2],
                gt_warped_ecc[evy1:evy2, evx1:evx2],
                cv2.TM_CCOEFF_NORMED)[0, 0])
        else:
            ncc_after = 0.0
        if ncc_after > ncc_score:
            T_refine = np.eye(3, dtype=np.float64)
            T_refine[0, 2] = dtx
            T_refine[1, 2] = dty
            H_roi = T_refine @ H_roi
            gt_warped = gt_warped_ecc
            _reg_debug += f"\n>> ECC refine: dtx={dtx:.2f} dty={dty:.2f} NCC {ncc_score:.4f}→{ncc_after:.4f}"
            ncc_score = ncc_after
        else:
            _reg_debug += f"\n>> ECC refine rejected: NCC {ncc_score:.4f}→{ncc_after:.4f} (worse)"
    except cv2.error as e:
        _reg_debug += f"\n>> ECC refine FAILED: {e} (using template-match only)"

    # Store match visualization for GUI
    _reg_match_img = _build_match_visualization(
        full_frame, tx, ty, tw, th, (px1, py1, px2, py2), label, ncc_match)

    # Cache registration
    _reg_data = {"H_roi": H_roi, "gt_rot": gt_oriented,
                 "ncc": ncc_score, "method": "TmplMatch",
                 "src_h": h, "src_w": w}

    # Apply beam offset and warp for this frame
    T_offset = np.eye(3, dtype=np.float64)
    T_offset[0, 2] = pixel_offset[0]
    T_offset[1, 2] = pixel_offset[1]
    H_final = T_offset @ H_roi
    gt_warped = cv2.warpPerspective(
        gt_resized, H_final, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)
    mask_warp = cv2.warpPerspective(
        np.ones((h, w), dtype=np.float32), H_final, (w, h)) > 0.5
    rows = np.any(mask_warp, axis=1)
    cols = np.any(mask_warp, axis=0)
    if not rows.any() or not cols.any():
        _reg_status = "Fallback"
        return None
    y1 = int(np.where(rows)[0][0]);  y2 = int(np.where(rows)[0][-1]) + 1
    x1 = int(np.where(cols)[0][0]);  x2 = int(np.where(cols)[0][-1]) + 1
    _reg_status = f"TmplMatch (ncc={ncc_score:.3f})"
    return gt_warped, (y1, y2, x1, x2), float(ncc_score)


_SNR_BORDER = 10  # pixels to exclude at each edge (SR algorithm produces a thin bezel)


def _compute_snr_pair(gt_crop: np.ndarray, img: np.ndarray, store_match: bool = False,
                      pixel_offset=(0.0, 0.0)):
    """Return (psnr, ssim, ncc_score) by registering gt_crop onto img.
    Uses template matching (full frame) + ECC MOTION_TRANSLATION (cached); falls back to discrete orientation.
    pixel_offset: sub-pixel (dx, dy) in native image pixels for beam-shifted sub-images.
    Applies a _SNR_BORDER-pixel inset to exclude the SR bezel before scoring.
    If store_match=True, saves the aligned GT crop to _reg_match_img for GUI display."""
    global _reg_match_img
    from skimage.metrics import peak_signal_noise_ratio as _psnr, structural_similarity as _ssim
    result = _register_gt(gt_crop, img, pixel_offset=pixel_offset,
                          store_match=store_match)
    ecc_score = float('nan')
    if result is not None:
        gt_warped, (y1, y2, x1, x2), ecc_score = result
        gt_aligned = gt_warped[y1:y2, x1:x2]
        img_eval = img[y1:y2, x1:x2]
    else:
        # Fallback: resize to img dims, then pick best discrete orientation
        gt_resized = cv2.resize(gt_crop, (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_AREA)
        gt_aligned = _best_orient(gt_resized, img)
        img_eval = img
    if store_match and result is None:
        _reg_match_img = gt_aligned  # fallback only; normal path handled by _register_gt
    # Exclude border pixels (SR bezel)
    b = _SNR_BORDER
    if gt_aligned.shape[0] > 2 * b and gt_aligned.shape[1] > 2 * b:
        gt_aligned = gt_aligned[b:-b, b:-b]
        img_eval = img_eval[b:-b, b:-b]
    min_dim = min(gt_aligned.shape[0], gt_aligned.shape[1])
    if min_dim < 7:
        return float('nan'), float('nan'), float(ecc_score)
    img_fit = _lin_norm(gt_aligned, img_eval)
    psnr_val = _psnr(gt_aligned, img_fit, data_range=1.0)
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    ssim_val = _ssim(gt_aligned, img_fit, data_range=1.0, win_size=win_size)
    return float(psnr_val), float(ssim_val), float(ecc_score)


def _save_filename(frame_type: str) -> Path:
    cam_str = {BASLER: "BASLER", DAHENG: "DAHENG_COLOR", DAHENG_MONO: "DAHENG_MONO"}[cameraType]
    roi = f"roi{ROI_width}x{ROI_height}"
    tilt = f"tilt{tilt_angle:.5f}"
    snr_tag = ""
    if snr_enabled and "xpr" in frame_type and not math.isnan(snr_xpr_psnr):
        snr_tag = f"_psnr{snr_xpr_psnr:.1f}dB_ssim{snr_xpr_ssim:.3f}"
    elif snr_enabled and "raw" in frame_type and not math.isnan(snr_avg_psnr):
        snr_tag = f"_psnravg{snr_avg_psnr:.1f}dB_ssim{snr_avg_ssim:.3f}"
    fname = f"{save_set_ts}_{cam_str}_exp{current_exposure}us_gain{current_gain}dB_{roi}_{tilt}{snr_tag}_{frame_type}.png"
    return save_dir / fname


class Window(QMainWindow, Ui_MainWindow):
    # signals corresponding to processes that happen in separate thread
    updatedCamImage = pyqtSignal()
    updatedCamZoom = pyqtSignal()
    updatedCamXPR = pyqtSignal()
    updatedFPS = pyqtSignal()
    updatedFPS2 = pyqtSignal()
    updatedSNR = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setupUi(self)  # load Qt created ui file (pyUIC autogenerated python file)
        self.updatedCamImage.connect(self.updateCamImage)
        self.updatedCamZoom.connect(self.updateCamZoom)
        self.updatedCamXPR.connect(self.updateCamXPR)
        self.updatedFPS.connect(self.updateFPS)
        self.updatedFPS2.connect(self.updateFPS2)

        # Set size of windows depending on the size of the size of the camera sensor
        self.CamImage.setMaximumSize(QtCore.QSize(size_h_max * w // h, size_h_max))
        self.CamImage.setMinimumSize(QtCore.QSize(size_h_max * w // h, size_h_max))
        self.CamZoom.setMaximumSize(QtCore.QSize(size_h_max * w // h, size_h_max))
        self.CamZoom.setMinimumSize(QtCore.QSize(size_h_max * w // h, size_h_max))
        self.CamXPR.setMaximumSize(QtCore.QSize(size_h_max * w // h, size_h_max))
        self.CamXPR.setMinimumSize(QtCore.QSize(size_h_max * w // h, size_h_max))

        # Set min/max values of the sliders depending on the size of the camera sensor
        self.RoiSize.setMinimum(ROI_width // 16)
        self.RoiSize.setMaximum(ROI_width * 4)
        self.RoiSize.setSliderPosition(ROI_width)

        if cameraType == BASLER:
            self.TitleImage.setText("Monochrome Camera")
            self.ButtonWBalance.hide()

            self.TitleChannel.hide()
            self.radioButton_RGB.hide()
            self.radioButton_R.hide()
            self.radioButton_G.hide()
            self.radioButton_B.hide()

            self.TitleHighlight.hide()
            self.radioButtonNormal.hide()
            self.radioButtonRed.hide()
            self.radioButtonBlue.hide()
            self.radioButtonRedBlue.hide()


        elif cameraType == DAHENG:
            self.TitleImage.setText("Color Camera")

            self.radioButton_Normal.hide()
            self.radioButton_Interpolated.hide()

        elif cameraType == DAHENG_MONO:
            self.TitleImage.setText("Monochrome Camera (Daheng)")
            self.ButtonWBalance.hide()

            self.TitleChannel.hide()
            self.radioButton_RGB.hide()
            self.radioButton_R.hide()
            self.radioButton_G.hide()
            self.radioButton_B.hide()

            self.TitleHighlight.hide()
            self.radioButtonNormal.hide()
            self.radioButtonRed.hide()
            self.radioButtonBlue.hide()
            self.radioButtonRedBlue.hide()

        # ---- Tilt Angle controls (added to right XPR column) ----
        self.TitleTiltAngle = QLabel("Tilt Angle (rad)")
        self.TiltAngleSlider = QSlider(Qt.Horizontal)
        self.TiltAngleSlider.setRange(0, 5000)
        self.TiltAngleSlider.setValue(round(tilt_angle * 10000))
        self.TiltAngleSlider.setTickInterval(100)
        self.TiltAngleSlider.setMaximumWidth(200)
        self.TiltAngleValue = QLabel(f"{tilt_angle:.5f}")

        tilt_preset_row = QHBoxLayout()
        self.ButtonTiltDahengColor = QPushButton("Daheng\nColor")
        self.ButtonTiltDahengMono = QPushButton("Daheng\nMono")
        self.ButtonTiltBasler = QPushButton("Basler")
        tilt_preset_row.addWidget(self.ButtonTiltDahengColor)
        tilt_preset_row.addWidget(self.ButtonTiltDahengMono)
        tilt_preset_row.addWidget(self.ButtonTiltBasler)

        self.XPRLayout.addWidget(self.TitleTiltAngle)
        self.XPRLayout.addWidget(self.TiltAngleSlider)
        self.XPRLayout.addWidget(self.TiltAngleValue)
        self.XPRLayout.addLayout(tilt_preset_row)

        self.TiltAngleSlider.valueChanged.connect(self.set_tilt_angle_from_slider)
        self.ButtonTiltDahengColor.clicked.connect(lambda: self.set_tilt_preset(0.14391 * 2))
        self.ButtonTiltDahengMono.clicked.connect(lambda: self.set_tilt_preset(0.14391))
        self.ButtonTiltBasler.clicked.connect(lambda: self.set_tilt_preset(0.05005))

        # ---- Save controls (added to middle Start/Stop column) ----
        self.ButtonSave = QPushButton("Start Saving")
        self.ButtonSave.setFixedSize(200, 40)
        self.SaveDirLabel = QLabel("(not saving)")
        self.SaveDirLabel.setWordWrap(True)
        self.SaveDirLabel.setMaximumWidth(200)

        self.StartStopLayout.addWidget(self.ButtonSave)
        self.StartStopLayout.addWidget(self.SaveDirLabel)

        self.ButtonSave.clicked.connect(self.toggle_save)

        # ---- Ground Truth controls (mini-box stays in right XPR column) ----
        snr_ctrl = QGroupBox("Ground Truth")
        snr_ctrl_inner = QVBoxLayout()

        self.ButtonLoadGT = QPushButton("Load GT Image\u2026")
        self.GTStatusLabel = QLabel("No GT loaded")
        self.GTStatusLabel.setWordWrap(True)
        self.ButtonEnableSNR = QCheckBox("Enable SNR")
        self.ButtonEnableSNR.setEnabled(False)
        self.ButtonReRegister = QPushButton("Re-register")
        self.ButtonReRegister.setEnabled(False)

        for _w in [self.ButtonLoadGT, self.GTStatusLabel,
                   self.ButtonEnableSNR, self.ButtonReRegister]:
            snr_ctrl_inner.addWidget(_w)
        snr_ctrl.setLayout(snr_ctrl_inner)
        self.XPRLayout.addWidget(snr_ctrl)

        # ---- Large GT Match & SNR display panel (inserted beside CamXPR in LowerLayout) ----
        snr_panel = QGroupBox("GT Match & SNR")
        snr_panel_inner = QVBoxLayout()

        self.GTMatchLabel = QLabel("GT match will appear here")
        self.GTMatchLabel.setMinimumSize(300, 225)
        self.GTMatchLabel.setAlignment(Qt.AlignCenter)
        self.GTMatchLabel.setStyleSheet("border: 1px solid gray; background: #111; color: gray;")

        self.SNRRegLabel = QLabel("Registration: \u2014")
        self.SNRSubLabels = [QLabel(f"Sub {_i}:  PSNR \u2014 dB  SSIM \u2014") for _i in range(4)]
        self.SNRAvgLabel  = QLabel("Avg:  PSNR \u2014 dB  SSIM \u2014")
        self.SNRXprLabel  = QLabel("XPR:  PSNR \u2014 dB  SSIM \u2014")
        self.SNRDeltaLabel = QLabel("\u0394 PSNR (XPR vs Avg)  \u2014")
        self.SNRInfoGainLabel = QLabel("\u0394 Info (XPR vs Avg)  \u2014")
        self.SNRDebugLabel = QLabel("")
        self.SNRDebugLabel.setWordWrap(True)
        self.SNRDebugLabel.setStyleSheet("color: #888; font-size: 10px; font-family: monospace;")

        snr_panel_inner.addWidget(self.GTMatchLabel)
        snr_panel_inner.addWidget(self.SNRRegLabel)
        for _lbl in self.SNRSubLabels:
            snr_panel_inner.addWidget(_lbl)
        snr_panel_inner.addWidget(self.SNRAvgLabel)
        snr_panel_inner.addWidget(self.SNRXprLabel)
        snr_panel_inner.addWidget(self.SNRDeltaLabel)
        snr_panel_inner.addWidget(self.SNRInfoGainLabel)
        snr_panel_inner.addWidget(self.SNRDebugLabel)
        snr_panel.setLayout(snr_panel_inner)

        # Place in top row alongside camera controls (avoids competing with image display columns)
        self.UpperLayout.addWidget(snr_panel)

        self.ButtonLoadGT.clicked.connect(self.load_gt_image)
        self.ButtonEnableSNR.toggled.connect(self.toggle_snr)
        self.ButtonReRegister.clicked.connect(self.invalidate_registration)
        self.updatedSNR.connect(self.updateSNR)

        self.CamImage.mousePressEvent = self.getPos

    def start_capture(self):
        global grab_mode
        mutex.lock()
        if (not grab_mode):
            pool = QThreadPool.globalInstance()  # we only use one thread in this pool
            # 2. Instantiate the subclass of QRunnable
            runnable = Runnable_Full_Cam()  # thread for image processing

            # 3. Call start()
            pool.start(runnable)
        mutex.unlock()

    def toggle_XPR(self):
        global XPR_on, frame_number
        XPR_on = not XPR_on
        _translate = QtCore.QCoreApplication.translate
        if XPR_on:
            self.ButtonXPR.setText(_translate("MainWindow", "Deactivate XPR"))
            self.TitleXPR.setText(_translate("MainWindow", "Image with XPR on"))
            mutex.lock()
            frame_number = 0
            mutex.unlock()
        else:
            self.ButtonXPR.setText(_translate("MainWindow", "Activate XPR"))
            self.TitleXPR.setText(_translate("MainWindow", "Image with XPR off"))
            mutex.lock()
            frame_number = 0
            mutex.unlock()
            # frame = cv2.cvtColor(np.zeros((1, 1, 1), dtype=np.uint8), cv2.COLOR_GRAY2RGB)
            # self.CamXPR.setPixmap(QPixmap(QImage(frame, 1, 1, 3, QImage.Format.Format_RGB888)))

    def stop_capture(self):
        global grab_mode, frame_number
        mutex.lock()
        grab_mode = False
        frame_number = 0
        mutex.unlock()

    def setWidth(self, value):
        global ROI_width, ROI_center_x, ROI_posx_min, ROI_posx_max
        global ROI_height, ROI_center_y, ROI_posy_min, ROI_posy_max
        global frame_number

        mutex.lock()  # lock the variables for the other thread

        ROI_width = value
        ROI_posx_min = value // 2 + w_line
        ROI_posx_max = w - value // 2 - w_line
        if ROI_center_x < value // 2 + w_line:
            ROI_center_x = value // 2 + w_line
        elif ROI_center_x > w - value // 2 - w_line:
            ROI_center_x = w - value // 2 - w_line

        ROI_height = int(h * value / w)
        ROI_posy_min = ROI_height // 2 + w_line
        ROI_posy_max = h - ROI_height // 2 - w_line
        if ROI_center_y < ROI_height // 2 + w_line:
            ROI_center_y = ROI_height // 2 + w_line
        elif ROI_center_y > h - ROI_height // 2 - w_line:
            ROI_center_y = h - ROI_height // 2 - w_line

        frame_number = 0

        mutex.unlock()

        self.Value_ROIWidth.setNum(value)
        self.Value_ROIHeight.setNum(int(h * value / w))

    def resetSizePos(self):
        global ROI_width, ROI_height, ROI_center_x, ROI_center_y, ROI_posx_min, ROI_posx_max, ROI_posy_min, ROI_posy_max
        mutex.lock()
        ROI_width = w // 8
        ROI_height = h // 8
        ROI_center_x = w // 2
        ROI_center_y = h // 2

        ROI_posx_min = ROI_width // 2 + w_line
        ROI_posx_max = w - ROI_width // 2 - w_line
        ROI_posy_min = ROI_height // 2 + w_line
        ROI_posy_max = h - ROI_height // 2 - w_line
        mutex.unlock()

        self.RoiSize.sliderPosition = w // 8
        self.RoiSize.setValue(w // 8)
        self.Value_ROIWidth.setNum(w // 8)
        self.Value_ROIHeight.setNum(h // 8)
        self.RoiSize.update()

    def updateCamImage(self):
        global frame_CamImage
        mutex.lock()
        self.CamImage.setPixmap(QPixmap(frame_CamImage))
        mutex.unlock()

    def updateCamZoom(self):
        global frame_CamZoom
        mutex.lock()
        self.CamZoom.setPixmap(QPixmap(frame_CamZoom))
        mutex.unlock()

    def updateCamXPR(self):
        global frame_CamXPR
        mutex.lock()
        self.CamXPR.setPixmap(QPixmap(frame_CamXPR))
        mutex.unlock()

    def updateFPS(self):
        global FPS
        mutex.lock()
        self.FPS.setText(str(round(FPS, 1)))
        mutex.unlock()

    def updateFPS2(self):
        global FPS2
        mutex.lock()
        self.FPS2.setText(str(round(FPS2, 1)))
        mutex.unlock()

    def setExposure(self, value):
        global current_exposure
        current_exposure = value
        if cameraType == BASLER:
            cam.ExposureTime.SetValue(value)
        else:
            cam.ExposureTime.set(value)

    def setGain(self, value):
        global current_gain
        current_gain = value
        if cameraType == BASLER:
            cam.Gain.SetValue(value)
        else:
            cam.Gain.set(value)

    def setAutoWhiteBalance(self):
        if cameraType == DAHENG and cam.BalanceWhiteAuto.is_writable():
            cam.BalanceWhiteAuto.set(2)

    def setAutoExposure(self):
        global current_exposure
        if cameraType in (DAHENG, DAHENG_MONO):
            cam.ExposureAuto.set(2)
            time.sleep(1)
            value = int(cam.ExposureTime.get())

        else:
            cam.ExposureAuto.SetValue("Continuous")
            time.sleep(1)
            cam.ExposureAuto.SetValue("Off")
            value = int(cam.ExposureTime.GetValue())

        current_exposure = value
        self.SliderExposure.sliderPosition = value
        self.SliderExposure.setValue(value)
        self.ExposureValue.setNum(value)
        self.SliderExposure.update()

    def setChannel(self):
        global channel
        mutex.lock()
        if self.radioButton_RGB.isChecked():
            channel = RGB
        elif self.radioButton_R.isChecked():
            channel = R
        elif self.radioButton_G.isChecked():
            channel = G
        elif self.radioButton_B.isChecked():
            channel = B
        mutex.unlock()

    def getPos(self, event):
        x = int(resize_factor * event.pos().x())
        y = int(resize_factor * event.pos().y())

        global ROI_width, ROI_height, ROI_center_x, ROI_center_y, ROI_posx_min, ROI_posx_max, ROI_posy_min, ROI_posy_max, frame_number

        mutex.lock()
        ROI_center_x = min(max(ROI_posx_min, x), ROI_posx_max)
        ROI_center_y = min(max(ROI_posy_min, y), ROI_posy_max)
        frame_number = 0
        mutex.unlock()

    def setInterpolationMode(self):
        global interpolation_mode
        mutex.lock()
        if self.radioButton_Normal.isChecked():
            interpolation_mode = NORMAL
        elif self.radioButton_Interpolated.isChecked():
            interpolation_mode = INTERPOLATED
        mutex.unlock()

    def setColorComparison(self):
        global color_comparison
        mutex.lock()
        if self.radioButtonNormal.isChecked():
            color_comparison = WHITE
        elif self.radioButtonRed.isChecked():
            color_comparison = RED
        elif self.radioButtonBlue.isChecked():
            color_comparison = BLUE
        elif self.radioButtonRedBlue.isChecked():
            color_comparison = REDBLUE
        mutex.unlock()

    def toggle_save(self):
        global save_enabled, save_dir
        save_enabled = not save_enabled
        if save_enabled:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path.home() / "xpr_saves" / ts
            save_dir.mkdir(parents=True, exist_ok=True)
            self.ButtonSave.setText("Stop Saving")
            self.SaveDirLabel.setText(f"~/xpr_saves/{ts}")
        else:
            self.ButtonSave.setText("Start Saving")
            self.SaveDirLabel.setText("(not saving)")

    def set_tilt_angle_from_slider(self, value):
        global tilt_angle, angles
        tilt_angle = value / 10000.0
        mutex.lock()
        angles[:] = tilt_angle * px_shifts
        mutex.unlock()
        self.TiltAngleValue.setText(f"{tilt_angle:.5f}")

    def set_tilt_preset(self, preset_val):
        global tilt_angle, angles
        tilt_angle = preset_val
        mutex.lock()
        angles[:] = tilt_angle * px_shifts
        mutex.unlock()
        self.TiltAngleSlider.setValue(round(tilt_angle * 10000))
        self.TiltAngleValue.setText(f"{tilt_angle:.5f}")

    def load_gt_image(self):
        global gt_original, gt_crop, gt_source_name
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Ground Truth Image", "",
            "Images & PDFs (*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.pdf)")
        if not path:
            return
        path = Path(path)

        if path.suffix.lower() == '.pdf':
            import fitz
            doc = fitz.open(str(path))
            page_num = 0
            if doc.page_count > 1:
                page_num, ok = QInputDialog.getInt(
                    self, "PDF Page",
                    f"Page number (0\u2013{doc.page_count - 1}):",
                    0, 0, doc.page_count - 1)
                if not ok:
                    return
            page = doc[page_num]
            mat = fitz.Matrix(600 / 72, 600 / 72)  # 600 DPI — 4× more pixels for better SIFT
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
            gt_img = arr.astype(np.float32) / 255.0
        else:
            img = cv2.imread(str(path))
            if img is None:
                QMessageBox.warning(self, "Load Error", f"Cannot load: {path.name}")
                return
            gt_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            # 4× nearest-neighbour upsample: barcodes are discrete, so NN preserves edges
            gt_img = cv2.resize(gt_img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

        dialog = GTPickerDialog(gt_img, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            gt_crop = dialog.get_crop()
            gt_original = gt_img
            gt_source_name = path.name
            self.GTStatusLabel.setText(
                f"GT: {path.name}\nCrop: {gt_crop.shape[1]}\u00d7{gt_crop.shape[0]} px")
            self.ButtonEnableSNR.setEnabled(True)
            self.invalidate_registration()
            self.ButtonReRegister.setEnabled(True)

    def toggle_snr(self, checked: bool):
        global snr_enabled
        snr_enabled = checked
        if not checked:
            for _i, _lbl in enumerate(self.SNRSubLabels):
                _lbl.setText(f"Sub {_i}:  PSNR \u2014 dB  SSIM \u2014")
            self.SNRAvgLabel.setText("Avg:  PSNR \u2014 dB  SSIM \u2014")
            self.SNRXprLabel.setText("XPR:  PSNR \u2014 dB  SSIM \u2014")
            self.SNRDeltaLabel.setText("\u0394 PSNR  \u2014")
            self.SNRInfoGainLabel.setText("\u0394 Info  \u2014")

    def invalidate_registration(self):
        global _reg_data, _sift_full_frame
        _reg_data = None
        _sift_full_frame = None
        self.SNRRegLabel.setText("Registration: (pending re-register)")
        self.SNRRegLabel.setStyleSheet("")

    def updateSNR(self):
        if math.isnan(snr_avg_psnr):
            return
        reg_color = "green" if "Fallback" not in _reg_status else "orange"
        self.SNRRegLabel.setText(f"Registration: {_reg_status}")
        self.SNRRegLabel.setStyleSheet(f"color: {reg_color};")
        for _i, _lbl in enumerate(self.SNRSubLabels):
            if not math.isnan(snr_sub_psnr[_i]):
                ncc_str = f"  NCC {snr_sub_ecc[_i]:.3f}" if not math.isnan(snr_sub_ecc[_i]) else ""
                _lbl.setText(f"Sub {_i}:  {snr_sub_psnr[_i]:.1f} dB  SSIM {snr_sub_ssim[_i]:.3f}{ncc_str}")
        avg_ncc_str = f"  NCC {snr_avg_ecc:.3f}" if not math.isnan(snr_avg_ecc) else ""
        self.SNRAvgLabel.setText(f"Avg:  {snr_avg_psnr:.1f} dB  SSIM {snr_avg_ssim:.3f}{avg_ncc_str}")
        xpr_ncc_str = f"  NCC {snr_xpr_ecc:.3f}" if not math.isnan(snr_xpr_ecc) else ""
        self.SNRXprLabel.setText(f"XPR:  {snr_xpr_psnr:.1f} dB  SSIM {snr_xpr_ssim:.3f}{xpr_ncc_str}")
        delta = snr_xpr_psnr - snr_avg_psnr
        sign = "+" if delta >= 0 else ""
        color = "green" if delta > 0 else "red"
        self.SNRDeltaLabel.setText(f"\u0394 PSNR (XPR vs Avg)  {sign}{delta:.1f} dB")
        self.SNRDeltaLabel.setStyleSheet(f"color: {color}; font-weight: bold;")
        # Information gain: same PSNR at higher resolution = more information
        info_gain = delta + 10 * math.log10(4)  # 2× resolution = 4× pixels → +6.02 dB
        ig_sign = "+" if info_gain >= 0 else ""
        ig_color = "green" if info_gain > 0 else "red"
        self.SNRInfoGainLabel.setText(f"\u0394 Info (XPR vs Avg)  {ig_sign}{info_gain:.1f} dB")
        self.SNRInfoGainLabel.setStyleSheet(f"color: {ig_color}; font-weight: bold;")
        if _reg_debug:
            self.SNRDebugLabel.setText(_reg_debug)
        if _reg_match_img is not None:
            m = _reg_match_img
            if m.ndim == 3:  # RGB — SIFT match visualization
                m_u8 = m if m.dtype == np.uint8 else (m * 255).astype(np.uint8)
                qimg = QImage(m_u8.data, m_u8.shape[1], m_u8.shape[0],
                              3 * m_u8.shape[1], QImage.Format_RGB888)
            else:  # Grayscale — aligned GT
                m_u8 = (m * 255).astype(np.uint8)
                qimg = QImage(m_u8.data, m_u8.shape[1], m_u8.shape[0],
                              m_u8.shape[1], QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg).scaled(
                self.GTMatchLabel.width(), self.GTMatchLabel.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.GTMatchLabel.setPixmap(pix)


app = QApplication(sys.argv)
win = Window()
win.start_capture()
win.setAutoExposure()
win.setAutoWhiteBalance()
win.showMaximized()
sys.exit(app.exec())
