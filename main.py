import numpy as np
import time
import win32gui
import pyautogui
import inputhandler
import joy
from trainer import make_large_cnn_pred, train_lstm_model, combine_test_data, combine_training_data, get_large_cnn_model
from joy import update_controller_positions


def get_controller_input_for_training():
    hwnd = win32gui.FindWindow(None, 'AC2  ')
    frame_zero = time.time()
    all_states = []
    all_training_data = []
    z1 = 0
    z2 = 0
    if hwnd != 0:
        x, y, x1, y1 = win32gui.GetClientRect(hwnd)
        x, y = win32gui.ClientToScreen(hwnd, (x, y))
        while True:
            while z1 < 1000:
                this_frame_time = time.time()  # start time of the loop
                im = pyautogui.screenshot(region=(x, y, x1, y1))
                img = im.resize(size=(256, 144))
                current_values = np.array(inputhandler.get_current_controllerstate(this_frame_time - frame_zero))
                trainingdata = np.array([this_frame_time - frame_zero, np.asarray(img)])
                all_states.append(current_values)
                all_training_data.append(trainingdata)
                z1 = z1+1
            print(z2)
            z1 = 0
            if z2 % 5 == 0:
                np.save('E:/ACCData/test_data/inputs' + str(z2), all_states)
                np.save('E:/ACCData/test_data/frames' + str(z2), all_training_data)
                print("Saved Iteration" + str(z2))
            else:
                np.save('E:/ACCData/training_data/inputs' + str(z2), all_states)
                np.save('E:/ACCData/training_data/frames' + str(z2), all_training_data)
                print("Saved Iteration" + str(z2))
            z2 += 1
            all_states = []
            all_training_data = []


def activate_ai(model):
    hwnd = win32gui.FindWindow(None, 'AC2  ')
    if hwnd != 0:
        x, y, x1, y1 = win32gui.GetClientRect(hwnd)
        x, y = win32gui.ClientToScreen(hwnd, (x, y))
        while True:
            im = pyautogui.screenshot(region=(x, y, x1, y1))
            img = im.resize(size=(256, 144))
            values = make_large_cnn_pred(model, img)
            update_controller_positions(values[0][0], values[0][1], values[0][2])

joy.reset_controller_pos()


#get_controller_input_for_training()
#combine_training_data()
#combine_test_data()

# model = get_large_cnn_model()


# train_lstm_model()
#activate_ai(model)

