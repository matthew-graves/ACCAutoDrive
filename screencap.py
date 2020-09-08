import numpy as np
import time
import win32gui
import pyautogui
import inputhandler
import joy
from PIL import Image
from trainer import make_large_cnn_pred, make_lstm_pred, get_large_cnn_model
from joy import update_controller_positions
# inputhandler.watch_controller()
model = get_large_cnn_model()

hwnd = win32gui.FindWindow(None, 'AC2  ')
frame_zero = time.time()
all_states = []
all_training_data = []
z1 = 0
z2 = 0
if hwnd != 0:
    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    w = right - left
    h = bot - top

    x, y, x1, y1 = win32gui.GetClientRect(hwnd)
    x, y = win32gui.ClientToScreen(hwnd, (x, y))
    while True:
        while z1 < 1000:
            this_frame_time = time.time()  # start time of the loop
            im = pyautogui.screenshot(region=(x, y, x1, y1))
            # im = im.convert('L')
            # img = Image.fromarray(np.asarray(im), 'L')
            # img = img.resize(size=(256, 144))
            img = im.resize(size=(256, 144))
            # width, height = img.size
            # img.show()
            # Setting the points for cropped image
            # Vleft = width - (width * 0.9)
            # top = height - (height * 0.75)
            # right = width * 0.9
            # bottom = height * 0.75

            # Cropped image of above dimension
            # (It will not change orginal image)
            # im1 = img.crop((left, top, right, bottom))
            values = make_large_cnn_pred(model, img)
            update_controller_positions(values[0][0], values[0][1], values[0][2])

            # current_values = np.array(inputhandler.get_current_controllerstate(this_frame_time - frame_zero))
            # trainingdata = np.array([this_frame_time - frame_zero, np.asarray(img)])
            # all_states.append(current_values)
            # all_training_data.append(trainingdata)
            # z1 = z1+1
        print(z2)
        z1 = 0
        # if z2 % 5 == 0:
        #     np.save('E:/ACCData/test_data/inputs' + str(z2), all_states)
        #     np.save('E:/ACCData/test_data/frames' + str(z2), all_training_data)
        #     print("Saved Iteration" + str(z2))
        # else:
        #     np.save('E:/ACCData/training_data/inputs' + str(z2), all_states)
        #     np.save('E:/ACCData/training_data/frames' + str(z2), all_training_data)
        #     print("Saved Iteration" + str(z2))
        z2 += 1
        all_states = []
        all_training_data = []
