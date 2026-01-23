import time
from zaber_motion import Units
from zaber_motion.ascii import Connection

with Connection.open_serial_port("/dev/tty.usbmodem1235541") as connection:
    connection.enable_alerts()

    device_list = connection.detect_devices()
    print("Found {} devices".format(len(device_list)))

    device = device_list[0]

    axis_count = device.axis_count

    print("There are ", axis_count, " axis on this device")

    y_axis = device.get_axis(3)
    z_axis = device.get_axis(4)
    x_axis = device.get_lockstep(1)


    # x_axis.home(True)
    # y_axis.home(True)
    # z_axis.home(True)


    y_axis_limit = y_axis.settings.get("limit.max", Units.LENGTH_CENTIMETRES)
    x_axis_limit_a = device.get_axis(1).settings.get("limit.max", Units.LENGTH_CENTIMETRES)
    x_axis_limit_b = device.get_axis(1).settings.get("limit.max", Units.LENGTH_CENTIMETRES)
    z_axis_limit = z_axis.settings.get("limit.max", Units.LENGTH_CENTIMETRES)


    x_axis.move_absolute(min(x_axis_limit_b, x_axis_limit_a)/2, Units.LENGTH_CENTIMETRES, True)
    y_axis.move_absolute(y_axis_limit/2, Units.LENGTH_CENTIMETRES, True)
    z_axis.move_absolute(z_axis_limit/2, Units.LENGTH_CENTIMETRES, True)

    x_axis.move_sin(10, Units.LENGTH_CENTIMETRES, 4, Units.TIME_SECONDS, 4, False)
    time.sleep(1)
    y_axis.move_sin(10, Units.LENGTH_CENTIMETRES, 4, Units.TIME_SECONDS, 4, False)

    time.sleep(5)

    # x_axis.home(True)
    # y_axis.home(True)
    # z_axis.home(True)

    print("x_axis_limit_a:", x_axis_limit_a)
    print("x_axis_limit_b:", x_axis_limit_b)
    print("y_limit:", y_axis_limit)
    print("z limt", z_axis_limit)
