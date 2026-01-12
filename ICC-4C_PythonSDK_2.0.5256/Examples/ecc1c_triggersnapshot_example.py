"""
Example how to set signal generator with trigger and save Snapshot on ECC-1C controller.
"""

import optoICC

# Setting Waveform parameters
amplitude_FP = 2  # Focal Power amplitude
frequency = 2  # Hz
# offset = 0.5
# phase = 0

#Connecting to board. Port can be specified like connect(port='COM18')
ecc1c = optoICC.connectEcc(port='COM18', baudrate=256000)
print("ECC-1C connected.")

lens = ecc1c.Lens
sign_generator = lens.SignalGenerator

# print(lens.Manager.CheckSignalFlow())  # Signal Flow Manager check

sign_generator.SetAsInput()
sign_generator.SetShape(optoICC.WaveformShape.SINUSOIDAL)
sign_generator.SetFrequency(frequency)
sign_generator.SetAmplitude(amplitude_FP)
# sign_generator.SetOffset(offset)
# sign_generator.SetPhase(phase)

# print(sign_generator.get_register('external_trigger'))  # check of value in 0x6009 register
# Can be synchronised with external signal (0=disabled, 1=rising/falling edge, 2=rising edge trigger)
sign_generator.SetExternalTrigger(2)

# print(sign_generator.GetExtTriggerGPIO())  # check of value in 0x600c register
# Turns GPIOx pin to trigger input source (GPIOx default is trigger output) and disables global external trigger SYNCx pin
sign_generator.SetExtTriggerGPIO(True)

sign_generator.SetCycles(1)  # 1 cycle, period is synchronized with trigger
sign_generator.SetUnit(optoICC.UnitType.FP)  # Unit diopters
# capacity = ecc1c.SnapshotManager.GetSnapShotCapacity()  # the capacity of ECC-1C is 1
# print(capacity)

ecc1c.save_snapshot(1)
ecc1c.SnapshotManager.SetDefaultSnapShot(1)
# ecc1c.load_snapshot(1)  # if it is not set default, it can be loaded

# To reset the driver back to factory settings
# ecc1c.SnapshotManager.SetDefaultSnapShot(0)
