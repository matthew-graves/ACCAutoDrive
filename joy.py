import pyvjoy
import math
import pdb

# Pythonic API, item-at-a-time
j = pyvjoy.VJoyDevice(1)

# Set X axis to fully left
j.set_axis(pyvjoy.HID_USAGE_X, 0x1)
# Set X axis to fully right
j.set_axis(pyvjoy.HID_USAGE_X, 0x8000)
#Set X axis to fully left
j.set_axis(pyvjoy.HID_USAGE_Y, 0x1)
#Set X axis to fully right
j.set_axis(pyvjoy.HID_USAGE_Y, 0x8000)
#Set X axis to fully left
j.set_axis(pyvjoy.HID_USAGE_Z, 0x1)
#Set X axis to fully right
j.set_axis(pyvjoy.HID_USAGE_Z, 0x8000)


def update_controller_positions(brake, gas, turning):
    if brake > gas:
        brake = 1
    else:
        gas = 1
    j.data.wAxisZ = math.ceil(gas * 32768)
    print(gas)
    print(brake)
    j.data.wAxisY = math.ceil(brake * 32768)
    turning *= 33000
    print(turning)
    j.data.wAxisX = math.ceil(turning)
    j.update()