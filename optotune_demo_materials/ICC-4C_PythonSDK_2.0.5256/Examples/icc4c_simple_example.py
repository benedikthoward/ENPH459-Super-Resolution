from optoICC import connect
# Connecting to board. Port can be specified like connect(port='COM12')
icc4c = connect()

# In this example lens is connected to channel 0
channel_index = 0

# In order to drive the connected product, device type of the product has to be obtained (this is not needed if
# automatic detection is allowed)
icc4c.MiscFeatures.GetDeviceType(channel_index)

# Each channel of the driver is controlled through its own Channel class
icc4c_lens_channel = icc4c.Channel_0

# Focal power will be set to 1 dpt
focal_power_value = 1

# To set the value specify the input type and set the value, driver will automatically switch to focal power mode
icc4c_lens_channel.StaticInput.SetAsInput()
icc4c_lens_channel.StaticInput.SetFocalPower(focal_power_value)
