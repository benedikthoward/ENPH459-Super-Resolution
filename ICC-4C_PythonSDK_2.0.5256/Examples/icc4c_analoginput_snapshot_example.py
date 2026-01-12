import optoICC

icc4c = optoICC.connect(inter_byte_timeout=0.5)  # longer timeout for snapshot manager
icc4c.reset(force=True)
icc4c.go_pro()

print("Connected")

icc4c.Channel_0.Analog.SetAsInput()
print("Analog Input Activated")
print("-----------------------")

# Use 0 for Current, 3 for Focal Power
icc4c.Channel_0.Analog.SetLUTtype(3)
print("Analog Input in FP mode")
print("-----------------------")

# Voltage values, up to 10 values in mapping  [V]
icc4c.Channel_0.Analog.SetLUTvoltages([0, 10])
print("Analog Input: Voltage values are saved")
print("-----------------------")

# Focal power values, up to 10 values in mapping [dpt]
icc4c.Channel_0.Analog.SetLUTvalues([-2, 3])
print("Analog Input: FP values are saved")
print("-----------------------")

icc4c.save_snapshot(2)
print("Analog Input: Snapshot saved")
print("-----------------------")

icc4c.SnapshotManager.SetDefaultSnapShot(2)
print("Analog Input: Default Snapshot Saved")
print("-----------------------")

print("Number of points in analog voltage input mapping:")
print(icc4c.Channel_0.Analog.GetLUTsize())
print("-----------------------")

print("Voltage values:")
print(icc4c.Channel_0.Analog.GetLUTvoltages())
print("-----------------------")

print("Focal power values:")
print(icc4c.Channel_0.Analog.GetLUTvalues())
print("-----------------------")

icc4c.disconnect()
print("Disconnected")