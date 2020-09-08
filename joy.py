import pyvjoy
import math
import pdb

# Pythonic API, item-at-a-time
j = pyvjoy.VJoyDevice(1)

def reset_controller_pos():

    # Set X axis to fully left
    j.set_axis(pyvjoy.HID_USAGE_X, 16384)
    # Set X axis to fully right
    # j.set_axis(pyvjoy.HID_USAGE_X, 32768)
    #Set X axis to fully left
    j.set_axis(pyvjoy.HID_USAGE_Y, 0x1)
    # Set X axis to fully right
    # j.set_axis(pyvjoy.HID_USAGE_Y, 32768)
    #Set X axis to fully left
    j.set_axis(pyvjoy.HID_USAGE_Z, 0x1)
    #Set X axis to fully right
    # j.set_axis(pyvjoy.HID_USAGE_Z, 32768)



def update_controller_positions(brake, gas, turning):
    #if brake > gas:
    #    brake = 0.5
    #else:
    #    gas = 0.5
    gas = math.ceil((16384 * (gas - 1) + 32768) * 0.8)
    brake = math.ceil((16384 * (brake - 1) + 32768) * 0.2)

    turning = (16384 * (turning - 1) + 32768)
    print(gas)
    print(brake)
    print(turning)
    j.data.wAxisZ = gas
    j.data.wAxisY = brake
    j.data.wAxisX = math.ceil(turning)
    j.update()