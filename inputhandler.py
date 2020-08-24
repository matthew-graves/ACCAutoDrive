import XInput as X
import numpy as np
import time
from pynput.keyboard import Listener




def get_current_controllerstate(current_time):

    inputdataset = np.zeros(4)
    state = X.get_state(0)
    triggers = X.get_trigger_values(state)
    joystick = X.get_thumb_values(state)[0][0]
    inputdataset[0] = current_time
    inputdataset[1] = triggers[0]
    inputdataset[2] = triggers[1]
    inputdataset[3] = joystick
    return inputdataset

