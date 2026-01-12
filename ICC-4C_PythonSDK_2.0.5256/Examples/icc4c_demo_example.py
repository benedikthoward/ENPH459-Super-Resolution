from time import sleep
from optoKummenberg import UnitType
from optoICC import connect, DeviceModel, WaveformShape

print("Starting basic optoICC example...")

#Connecting to board. Port can be specified like connect(port='COM12')
icc4c = connect()

print()
print("Board info")
#Getting board info
serial_number = icc4c.EEPROM.GetSerialNumber().decode('UTF-8')
fw_version = f"{icc4c.Status.GetFirmwareVersionMajor()[0]}.{icc4c.Status.GetFirmwareVersionMinor()[0]}.{icc4c.Status.GetFirmwareVersionRevision()[0]}"

print(f"Board serial number: {serial_number}")
print(f"Board firmware version: {fw_version}")

# In order to drive the connected product, device type of the product has to be obtained (this is not needed if
# automatic detection is allowed)
connected_devices = [icc4c.MiscFeatures.GetDeviceType(0), icc4c.MiscFeatures.GetDeviceType(1),
                     icc4c.MiscFeatures.GetDeviceType(2), icc4c.MiscFeatures.GetDeviceType(3)]
print(f"Connected devices: {''.join('{},'.format(x.name) for x in connected_devices)}")
print()
print("Driving lens")
first_connected_lens = None
for idx, device in enumerate(connected_devices):
    if device.value in [DeviceModel.EL_1030_C, DeviceModel.EL_1030_TC, DeviceModel.EL_1230_TC, DeviceModel.EL_1640_TC]:
        first_connected_lens = (device, idx)
        break

if first_connected_lens is None:
    print("No lens connected")
else:
    # Each channel can also be accessed directly as icc4c.Channel_0 instead of icc4c.channel[0]
    icc4c_lens_channel = icc4c.channel[first_connected_lens[1]]
    lens_serial_number = icc4c_lens_channel.DeviceEEPROM.GetSerialNumber().decode('UTF-8')
    print(f"Lens {first_connected_lens[0].name} ({lens_serial_number}) found on channel {first_connected_lens[1]}")

    lens_temperature = icc4c_lens_channel.TemperatureManager.GetDeviceTemperature()[0]
    print(f"Lens temperature: {lens_temperature} °C")

    min_lens_current = -(icc4c_lens_channel.DeviceEEPROM.GetMaxNegCurrent()[0])
    max_lens_current = icc4c_lens_channel.DeviceEEPROM.GetMaxPosCurrent()[0]
    print(f"Minimum current is {min_lens_current} mA, maximum current is {max_lens_current} mA")

    print()
    print(f"Setting static current")
    #Setting input system to static input
    icc4c_lens_channel.StaticInput.SetAsInput()

    for current in range(int(min_lens_current), int(max_lens_current)+1, int((max_lens_current-min_lens_current)/5)):
        #Value has to be converted from mA to A
        current_in_A = float(current)/1000
        print(f"Current {current_in_A} A")
        icc4c_lens_channel.StaticInput.SetCurrent(current_in_A)
        sleep(1)

    print("Setting static current to 0 A")
    icc4c_lens_channel.StaticInput.SetCurrent(0.0)
    print(" ")
    print("Running signal generator")
    icc4c_lens_channel.SignalGenerator.SetAsInput()
    icc4c_lens_channel.SignalGenerator.SetUnit(UnitType.CURRENT)
    icc4c_lens_channel.SignalGenerator.SetShape(WaveformShape.SINUSOIDAL)
    icc4c_lens_channel.SignalGenerator.SetAmplitude(0.2)
    icc4c_lens_channel.SignalGenerator.SetFrequency(5)
    icc4c_lens_channel.SignalGenerator.Run()

    for index in range(5):
        print(".", end="")
        sleep(1)

    icc4c_lens_channel.SignalGenerator.Stop()

    print()
    print("Signal generator stopped")

    print()
    print("Device EEPROM")

    eeprom_version = f"{icc4c_lens_channel.DeviceEEPROM.GetEEPROMversion()[0]}.{icc4c_lens_channel.DeviceEEPROM.GetEEPROMsubversion()[0]}"
    print(f"EEPROM version: {eeprom_version}")

    eeprom_bytes = icc4c_lens_channel.DeviceEEPROM.GetEEPROM(0, 10)
    eeprom_size = icc4c_lens_channel.DeviceEEPROM.GetEEPROMSize()[0]
    print(f"Printing {len(eeprom_bytes)}/{eeprom_size} bytes saved in EEPROM: {''.join('{:02x},'.format(x) for x in eeprom_bytes)}")

print()
print("Example finished.")
