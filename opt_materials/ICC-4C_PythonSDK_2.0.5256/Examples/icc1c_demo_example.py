from time import sleep
from optoKummenberg import UnitType
from optoICC import connectIcc1c, WaveformShape

print("Starting basic optoICC example...")

#Connecting to board. Port can be specified like connect(port='COM12')
icc1c = connectIcc1c()

print()
print("Board info")
#Getting board info
serial_number = icc1c.EEPROM.GetSerialNumber().decode('UTF-8')
fw_version = f"{icc1c.Status.GetFirmwareVersionMajor()}.{icc1c.Status.GetFirmwareVersionMinor()}.{icc1c.Status.GetFirmwareVersionRevision()}"

print(f"Board serial number: {serial_number}")
print(f"Board firmware version: {fw_version}")

connected_lens = icc1c.MiscFeatures.GetDeviceType(0)

if connected_lens is None:
    print("No lens connected")
else:
    lens_serial_number = icc1c.channel[0].DeviceEEPROM.GetSerialNumber().decode('UTF-8')
    print(f"Lens {connected_lens.name} ({lens_serial_number}) found.")

    lens_temperature = icc1c.TemperatureManager.GetDeviceTemperature()
    print(f"Lens temperature: {lens_temperature} °C")

    min_lens_current = -(icc1c.channel[0].DeviceEEPROM.GetMaxNegCurrent())
    max_lens_current = icc1c.channel[0].DeviceEEPROM.GetMaxPosCurrent()
    print(f"Minimum current is {min_lens_current} mA, maximum current is {max_lens_current} mA")

    print()
    print(f"Setting static current")
    print(icc1c.channel[0])
    #Setting input system to static input
    icc1c.channel[0].StaticInput.SetAsInput()

    for current in range(int(min_lens_current), int(max_lens_current)+1, int((max_lens_current-min_lens_current)/5)):
        #Value has to be converted from mA to A
        current_in_A = float(current)/1000
        print(f"Current {current_in_A} A")
        icc1c.channel[0].StaticInput.SetCurrent(current_in_A)
        sleep(1)

    print("Setting static current to 0 A")
    icc1c.channel[0].StaticInput.SetCurrent(0.0)
    print(" ")
    print("Running signal generator")
    icc1c.channel[0].SignalGenerator.SetAsInput()
    icc1c.channel[0].SignalGenerator.SetUnit(UnitType.CURRENT)
    icc1c.channel[0].SignalGenerator.SetShape(WaveformShape.SINUSOIDAL)
    icc1c.channel[0].SignalGenerator.SetAmplitude(0.2)
    icc1c.channel[0].SignalGenerator.SetFrequency(5)
    icc1c.channel[0].SignalGenerator.Run()

    for index in range(5):
        print(".", end="")
        sleep(1)

    icc1c.channel[0].SignalGenerator.Stop()

    print()
    print("Signal generator stopped")

    print()
    print("Device EEPROM")

    eeprom_version = f"{icc1c.channel[0].DeviceEEPROM.GetEEPROMversion()}.{icc1c.channel[0].DeviceEEPROM.GetEEPROMsubversion()}"
    print(f"EEPROM version: {eeprom_version}")

    eeprom_bytes = icc1c.channel[0].DeviceEEPROM.GetEEPROM(0, 10)
    eeprom_size = icc1c.channel[0].DeviceEEPROM.GetEEPROMSize()
    print(f"Printing {len(eeprom_bytes)}/{eeprom_size} bytes saved in EEPROM: {''.join('{:02x},'.format(x) for x in eeprom_bytes)}")

print()
print("Example finished.")
