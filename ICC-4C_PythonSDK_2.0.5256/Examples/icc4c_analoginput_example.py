from optoKummenberg import UnitType
from optoICC import connect, DeviceModel

print("Starting analog input optoICC example...")

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
print("Searching for lens")
first_connected_lens = None
for idx, device in enumerate(connected_devices):
    if device.value in [DeviceModel.EL_1030_C, DeviceModel.EL_1030_TC, DeviceModel.EL_1230_TC, DeviceModel.EL_1640_TC]:
        first_connected_lens = (device, idx)
        break

if first_connected_lens is None:
    print("No lens connected")
else:
    print(f"Lens found on channel {first_connected_lens[1]}")
    # Each channel can also be accessed directly as icc4c.Channel_0 instead of icc4c.channel[0]
    icc4c_lens_channel = icc4c.channel[first_connected_lens[1]]
    # Set Analog lookup table unit
    icc4c_lens_channel.Analog.SetUnitType_LUT(UnitType.FP)
    # Get Maximum Diopter and Minimum Diopter
    max_diopter = icc4c_lens_channel.LensCompensation.GetMaxDiopter()
    min_diopter = icc4c_lens_channel.LensCompensation.GetMinDiopter()

    voltages = icc4c_lens_channel.Analog.GetVoltages_LUT()
    values = icc4c_lens_channel.Analog.GetValues_LUT()

    # Specify the analog lookup table
    icc4c_lens_channel.Analog.SetVoltages_LUT([0, 10])
    icc4c_lens_channel.Analog.SetValues_LUT([min_diopter, max_diopter])
    icc4c_lens_channel.Analog.SetAsInput()

# Save settings to be available after restarting the driver
icc4c.SnapshotManager.SaveSnapshot(1)
icc4c.SnapshotManager.SetDefaultSnapShot(1)

# To reset default driver settings, set default snapshot number to 0
#icc4c.SnapshotManager.SetDefaultSnapShot(0)

