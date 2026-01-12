"""
Example how to set analog input signal with linear mapping for -2 to 3 dpts in
0 to 10 V range and save Snapshot on ECC-1C controller.
"""

import optoICC

#Connecting to board. Port can be specified like connect(port='COM4')
ecc1c = optoICC.connectEcc(port='COM4', verbose = True)
print("Connected to ECC-1C.")

lens = ecc1c.Lens

lens.Analog.SetAsInput()
print("Analog Input Activated")
print("-----------------------")

# Use 0 for Current, 3 for Focal Power
lens.Analog.SetLUTtype(3)
print("Analog Input in FP mode")
print("-----------------------")

# Min and Max input  [V]
lens.Analog.set_voltage_LUT([0, 10])
print("Analog Input: Voltage values are saved")
print("-----------------------")

# Min and Max focal power values [dpt]
lens.Analog.set_value_LUT([-2, 3])
print("Analog Input: FP values are saved")
print("-----------------------")

ecc1c.save_snapshot(1)
print("Analog Input: Snapshot saved")
print("-----------------------")

ecc1c.SnapshotManager.SetDefaultSnapShot(1)
print("Analog Input: Default Snapshot Saved")
print("-----------------------")

ecc1c.disconnect()
print("Disconnected")