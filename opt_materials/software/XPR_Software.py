import logging
import math
import cv2
import time
import optoICC
import sys
import numpy as np
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
cameraType = 0  # 0: DAHENG Color Camera, 1: BASLER Mono Camera
DAHENG = 0
BASLER = 1
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

    tilt_angle = 0.14391  # +/- half pixel shift for 3.45um pixels
   #tilt_angle = 0.1
    cameraType = DAHENG
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

# Depends on the Bayer pattern of the camera
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

        grab_mode = True
        if cameraType == BASLER:
            cam.StartGrabbing(1)

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
            else: # Daheng
                cam.TriggerSoftware.send_command()
                raw_image = cam.data_stream[0].get_image()
                rgb_image = raw_image.convert("RGB", convert_type=0)
                frame = rgb_image.get_numpy_array()

            bytesPerLine = 3 * w

            px1 = int(ROI_center_x - ROI_width // 2)
            px2 = int(ROI_center_x + ROI_width // 2)
            py1 = int(ROI_center_y - ROI_height // 2)
            py2 = int(ROI_center_y + ROI_height // 2)
            frameZoom = frame[py1:py2, px1:px2].copy()
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
                if np.array_equal(color_comparison, WHITE) or cameraType == BASLER:
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
                if cameraType == DAHENG:
                    frame_raw = raw_image.get_numpy_array()
                    height, width = frame_raw[py1: py2, px1: px2].shape

                    mutex.lock()
                    # enforce correct
                    if frame_CamXPR_numpy.shape[0] != frameZoom_resized.shape[0] or frame_CamXPR_numpy.shape[1] != \
                            frameZoom_resized.shape[1] or frame_number == 0:
                        frame_CamXPR_numpy = np.zeros((frameZoom_resized.shape[0], frameZoom_resized.shape[1], 3),
                                                      dtype=np.uint8)
                    mutex.unlock()

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

                    frame_CamXPR_numpy[::2, ::2, frame_number] = frame_raw[py1: py2, px1: px2]
                    frame_CamXPR_numpy[:, :, frame_number] = cv2.warpAffine(frame_CamXPR_numpy[:, :, frame_number],
                                                                            M[frame_number], (2 * width, 2 * height),
                                                                            borderMode=cv2.BORDER_REFLECT_101)

                    if frame_number == n_images - 1:
                        frame_CamXPR_numpy_HR = np.sum(frame_CamXPR_numpy, axis=2, dtype=np.uint8)
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


class Window(QMainWindow, Ui_MainWindow):
    # signals corresponding to processes that happen in separate thread
    updatedCamImage = pyqtSignal()
    updatedCamZoom = pyqtSignal()
    updatedCamXPR = pyqtSignal()
    updatedFPS = pyqtSignal()
    updatedFPS2 = pyqtSignal()

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
        if cameraType == BASLER:
            cam.ExposureTime.SetValue(value)
        else:
            cam.ExposureTime.set(value)

    def setGain(self, value):
        if cameraType == BASLER:
            cam.Gain.SetValue(value)
        else:
            cam.Gain.set(value)

    def setAutoWhiteBalance(self):
        if cameraType == DAHENG and cam.BalanceWhiteAuto.is_writable():
            cam.BalanceWhiteAuto.set(2)

    def setAutoExposure(self):
        if cameraType == DAHENG:
            cam.ExposureAuto.set(2)
            time.sleep(1)
            value = int(cam.ExposureTime.get())

        else:
            cam.ExposureAuto.SetValue("Continuous")
            time.sleep(1)
            cam.ExposureAuto.SetValue("Off")
            value = int(cam.ExposureTime.GetValue())

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


app = QApplication(sys.argv)
win = Window()
win.start_capture()
win.setAutoExposure()
win.setAutoWhiteBalance()
win.showMaximized()
sys.exit(app.exec())
