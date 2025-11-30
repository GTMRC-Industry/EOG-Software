import sys
import serial
import matplotlib.pyplot as plt
import numpy as np
from pynput import keyboard
from pynput.keyboard import Key, Controller

from collections import deque
import tkinter as tk
import random
import subprocess
import json
import os
from sklearn.linear_model import LogisticRegression, LinearRegression, HuberRegressor, RANSACRegressor
from scipy.signal import savgol_filter
import time

# Set up the serial port and parameters
serial_port = '/dev/cu.usbmodem11401'  # Replace with your Arduino's serial port (e.g., '/dev/cu.usbmodem101' on Linux or 'COM3' on Windows)
baud_rate = 230400
ser = serial.Serial(serial_port, baud_rate)
history = 500
update_batch_size = 20
update_cursor_batch_size = 0
a = 0.5 # Exponential Smoothing Parameter
baseline = 1023 / 2

# Plot Setup
# plt.ion()
# fig, ax = plt.subplots()
x_data = deque(maxlen=history)
y_data = deque(maxlen=history)
y_pred_deque = deque(maxlen=history)
# line, = ax.plot([], [], 'r-')
# ax.set_xlim(0, history)
# ax.set_ylim(0, 1000)
y_data_list = []
y_data_arr = []
blink_data_list = []
blink_data_arr = []

history_dY = []
history_dEOG = []

blink_cooldown = []
rt = np.array([])

after_blink = 0
#Feature Extraction Setup
global c 
c=0
global open_blink_calibration
global record_blink
global open_eye_mvmt_calibration
global record_eye_mvmt
global close_program
global d_count

open_blink_calibration = False
record_blink = False
open_eye_mvmt_calibration = False
record_eye_mvmt = False
close_program = False
d_count = 0


pending_blink_pause = False


print("BEFORE LISTENER")
def on_press(key):
    try:
        global open_blink_calibration
        global record_blink
        global open_eye_mvmt_calibration
        global record_eye_mvmt
        global close_program
        global d_count
        open_blink_calibration = False
        record_blink = False
        open_eye_mvmt_calibration = False
        record_eye_mvmt = False
        close_program = False
        if key.char == 'k':
            open_blink_calibration = True
        elif key.char == 'b':
            record_blink = True
            print('pressed b')
        elif key.char == 'c':
            open_eye_mvmt_calibration = True
        elif key.char == 'd':
            print('pressed d')
            d_count += 1
            record_eye_mvmt = True
        elif key.char == 'q':
            close_program = True
    except AttributeError:
        # handle special keys (e.g. space, enter)
        pass
listener = keyboard.Listener(on_press=on_press)
listener.start()

# EXTRACT FEATURES FOR TRAINING REGRESSION ------------------------

def get_deltaY(points):
    y_points = np.array([])
    for point in points: 
        y_points = np.append(y_points, point[1])

    #Apply first differences to y_points
    deltaY = np.diff(y_points)
    return deltaY

def get_deltaEOG_train(y_data_arr):
    deltaEOG_v = np.array([])
    # Sliding window
    window_size = 150
    for entry in y_data_arr:

        entry = savgol_filter(entry, 31, 3)   
        entry = entry - np.mean(entry[:10])

        i = 0
        window_deltas_y = []
        if max(entry, key=abs) > entry[0]:
            for i in range(len(entry) - window_size):
                window_deltas_y.append(max(entry[i: i + window_size]) - min(entry[i:i+ window_size]))
        else:
            for i in range(len(entry) - window_size):
                window_deltas_y.append(min(entry[i: i + window_size]) - max(entry[i:i+ window_size]))

        deltaEOG_v = np.append(deltaEOG_v, max(window_deltas_y, key=abs))

    return deltaEOG_v


def filter_data(deltaEOG_v, deltaY):
    if len(deltaEOG_v)==len(deltaY):
        model_v = RANSACRegressor(residual_threshold=5.0)
        print("Before fitting")
        model_v.fit(deltaEOG_v.reshape(-1, 1), deltaY)
        
        print("After fitting")
        inlier_mask = model_v.inlier_mask_
        deltaEOG_v = deltaEOG_v[inlier_mask]
        deltaY = deltaY[inlier_mask]

        return deltaEOG_v, deltaY
    else:
        print("deltaEOG_v len:", len(deltaEOG_v), "std:", np.std(deltaEOG_v))
        print("deltaY len:", len(deltaY), "std:", np.std(deltaY))
        sys.stdout.flush()

def train_model(deltaEOG_v, deltaY):
    model = LinearRegression()
    model.fit(deltaEOG_v.reshape(-1, 1), deltaY)
    # Save the trained model coefficients for later use
    with open("linear_model_coefficients.json", "w") as f:
        json.dump({"slope": model.coef_[0], "intercept": model.intercept_}, f)
    print("Model trained and coefficients saved.")
    return model


# BLINK CALIBRATION -------------------

def set_blink_threshold(blink_data):
    deltaEOG_blinks = np.array([])
    slopes = np.array([])
    accels = np.array([])
    # use sliding window to get the max deltaEOG, then return the min of those max deltaEOG's as our threshold.
    window_size = 200
    for entry in blink_data:

        entry = savgol_filter(entry, 31, 3)
        v = savgol_filter(entry, 31, 3, deriv=1)
        a = savgol_filter(entry, 31, 3, deriv=2)

        argmax = np.argmax(entry)
        entry = entry[:argmax]
        v = v[:argmax]
        a = a[:argmax]

        i = 0
        window_deltas_y = []
        
        for i in range(len(entry) - window_size):
            # signal went down first, then up → positive saccade
            window_deltas_y.append(max(entry[i: i + window_size]) - min(entry[i:i+ window_size]))

        max_diff_loc = np.argmax(window_deltas_y)
        max_window_bounds = (max_diff_loc, max_diff_loc + window_size)

        deltaEOG_blinks = np.append(deltaEOG_blinks, max(window_deltas_y, key=abs))
        slopes = np.append(slopes, max(v[max_window_bounds[0] : max_window_bounds[1]]))
        accels = np.append(accels, max(a[max_window_bounds[0] : max_window_bounds[1]]))


    
    deltaEOG_blink_std = np.std(deltaEOG_blinks)
    slope_std = np.std(slopes)
    deltaEOG_blink_lowest = np.min(deltaEOG_blinks) - deltaEOG_blink_std
    slope_lowest = np.min(slopes) - slope_std
    accel_lowest = np.mean(accels)

    return deltaEOG_blink_lowest, slope_lowest, accel_lowest


# CLASSIFY BLINK OR SACCADE ------------------k

def classify(deltaEOG_v, deltaEOG_blink_lowest, slope, slope_lowest, accel, accel_lowest):
    if  deltaEOG_v > deltaEOG_blink_lowest:
        blink = True

    else:
        blink = False

    return blink


def classify_only_peak_threshold(deltaEOG_v, deltaEOG_blink_lowest):
    if deltaEOG_v < deltaEOG_blink_lowest:
        blink = False
    else:
        blink = True

    return blink



# RUN REGRESSION ---------------------
def get_deltaEOG_test(y_data):
    deltaEOG_v = np.array([])
    # Sliding window
    window_size = 150
    i = 0
    window_deltas_y = []
    entry=list(y_data)
    entry = entry - np.mean(entry[:10])


    if max(entry, key=abs) > entry[0]: #if positive saccade 

        entry = savgol_filter(entry, 31, 3)
        v = savgol_filter(entry, 31, 3, deriv=1)
        a = savgol_filter(entry, 31, 3, deriv=2)

        argmax = np.argmax(entry)
        entry = entry[:argmax]
        v = v[:argmax]
        a = a[:argmax]

        for i in range(len(entry) - window_size):
            # signal went down first, then up → positive saccadekc
            window_deltas_y.append(max(entry[i: i + window_size]) - min(entry[i:i+ window_size]))

        max_diff_loc = np.argmax(window_deltas_y)
        max_window_bounds = (max_diff_loc, max_diff_loc + window_size)

        slope = max(v[max_window_bounds[0] : max_window_bounds[1]])
        accel = max(a[max_window_bounds[0] : max_window_bounds[1]])
    else:
        for i in range(len(entry) - window_size):
            # signal went up first, then down → negative saccade
            window_deltas_y.append(min(entry[i: i + window_size]) - max(entry[i:i+ window_size]))
        slope = -999
        accel = -999

    deltaEOG_v = np.append(deltaEOG_v, max(window_deltas_y, key=abs))

    return deltaEOG_v, slope, accel

calibrated = False
calibrating = False 
press=False
p = None

blink_calibrating = False
blink_calibrated = False
blink_press = False
n = None


def update_plot():
    line.set_xdata(np.arange(len(x_data)))  # X-axis
    line.set_ydata(np.array(y_data))  # Y-axis
    plt.draw()

def update_point_plot():
    line.set_xdata(np.arange(len(x_data)))
    line.set_ydata(np.array(y_pred_deque))
    plt.draw()
# Real-time plotting

### CURSOR SETUP

import pyautogui
from pynput.mouse import Controller, Button
mouse = Controller()
import time

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

SCALE = 1   # multiplier to turn deltaY into pixels
SMOOTHING = 0.6         # 0=no smoothing, 1=very smooth
FREEZE_ON_BLINK = True   # optional: freezes cursor during blink
MIN_MOVE_THRESHOLD = 10 # ignore tiny jitters

# ---------------------------------------------------------
# STATE VARIABLES
# ---------------------------------------------------------

screen_w, screen_h = pyautogui.size()

cursor_x = screen_w // 2
cursor_y = screen_h // 2

print(cursor_x, cursor_y)
mouse.position = (cursor_x, cursor_y)  # start centered


# ---------------------------------------------------------
# CURSOR UPDATE FUNCTION
# ---------------------------------------------------------

def update_cursor(deltaY):

    # if abs(deltaY) < MIN_MOVE_THRESHOLD:
    #     return  # ignokcre tiny movements
    
    # smoothed_delta = (1 - SMOOTHING) * deltaY
    # smoothed_delta = smoothed_delta * SCALE
    mouse.move(0, deltaY)
    # time.sleep(0.0005)










## MAIN LOOP
readings = 0
while True:
    if ser.in_waiting > 0:
        try:
            data = ser.readline().decode('utf-8').strip().split(',')[0]  # Read and decode data from serial
            value = int(data)
            # if y_data:
            #     value = value * a + (1 - a) * y_data[-1]
            x_data.append(len(x_data))
            y_data.append(value)

            # Blink Calibration phase
            if open_blink_calibration and not blink_calibrating:
                 # press k to open the blink game
                print('Opening Blink Game')
                n = subprocess.Popen(["python3", "blink_game.py"])
                blink_calibrating = True
            if blink_calibrating:

                blink_data_list.append(value)

                if record_blink:

                    if (not blink_press):
                        blink_press = True
                        record_blink = False
                        blink_current_idx = len(blink_data_list) - 1
                        print('Captured Blink')
                        blink_data_arr.append(blink_data_list[:blink_current_idx])
                        blink_data_list = []
                else:
                    blink_press = False

                if n.poll() is not None:

                    print('Blink Calibration Complete, Setting Blink Thresholds')

                    with open('raw_blinks.json', "w") as f:
                        json.dump(blink_data_arr, f)

                    deltaEOG_blink_lowest, slope_lowest, accel_lowest = set_blink_threshold(blink_data_arr)
                    print('blink deltaEOG threshold' , deltaEOG_blink_lowest)
                    print('blink slope threshold' , slope_lowest)
                    print('blink accel threshold' , accel_lowest)
                    blink_calibrating = False
                    blink_calibrated = True
                    print(blink_calibrating)
            


            # Eye Movement Calibration phase
            if open_eye_mvmt_calibration and not calibrating: # press c to open the calibration game
                print('Opening Calibration Game')
                p = subprocess.Popen(["python3", "updated_calibration_game.py"]) #run the calibration game
                calibrating = True
            
            if calibrating and d_count > 0:
                y_data_list.append(value) # while the game is being run...
                if record_eye_mvmt: # collect calibration data
                    if (not press):
                        press=True
                        record_eye_mvmt = False
                        current_idx = len(y_data_list) - 1
                        print('Captured Eye Movement')
                        # print(y_data_list)
                        y_data_arr.append(y_data_list[:current_idx])
                        y_data_list = []
                else:
                    press=False

                

                if p.poll() is not None:# when calibration is over
                    with open("calibration_points.json") as f: # load the list of random points generated in calibration_game
                        points = json.load(f)
                    # print('EYE MOVEMENT Calibration Complete.')
                    
                    y_data_arr.pop(0)

                    with open("raw_eye_movements.json", "w") as f:
                            json.dump(y_data_arr, f)


                    #TEST THESE FUNCITONS TODAY
                    # for prev_readings in y_data_arr:
                    #          print(len(prev_readings))
                    deltaEOG_v, deltaY = get_deltaEOG_train(y_data_arr), get_deltaY(points)

                    # dump json files of the recorded eye movements and points for future investigation
                    with open("deltaEOG_v_eye_movements.json", "w") as f:
                        json.dump(deltaEOG_v.tolist(), f)

                    with open("deltaY.json", "w") as f:
                        json.dump(deltaY.tolist(), f)
                    


                    #This filters out any blinks that may have occurred during calibration
                    filtered_deltaEOG_v = deltaEOG_v[[not classify_only_peak_threshold(entry, deltaEOG_blink_lowest) for entry in deltaEOG_v]] # filter out blinks that occurred in the eye movement calibration
                    deltaY = deltaY[[not classify_only_peak_threshold(entry, deltaEOG_blink_lowest) for entry in deltaEOG_v]] # align points with the newly filtered out eye movements
                    print('new length of deltaY: ', len(deltaY))


                    # deltaEOG_v, deltaY = filter_data(deltaEOG_v, deltaY)
                    # print("Filtering data")
                    print('before training')
                    model=train_model(filtered_deltaEOG_v, deltaY)
                    print('after training')
                    update_batch_size = 40


                    calibrating = False # stop the loop
                    calibrated = True
                    print(calibrating)
            #End Calibration phasekcq

            readings += 1

            #consider using separate batch sizes to deal with plot lag
            if (readings > update_batch_size) and (not calibrating) and (not blink_calibrating):
                # update_plot()
                #Call Model.predict here

                if calibrated and blink_calibrated: # if calibration finished

                    n = 3 # number of readings to delete before the blink is registered

                    deltaEOG_v, slope, accel = get_deltaEOG_test(y_data)

                    # update_cursor(0)
                    after_blink -= 1

                    if after_blink < 0:
                        history_dEOG.append(deltaEOG_v)

                        classification = classify(deltaEOG_v, deltaEOG_blink_lowest, slope, slope_lowest, accel, accel_lowest)

                        if not classification:
                            if abs(deltaEOG_v) < 30:
                                 y_pred = 0
                            else:
                                y_pred = model.predict(deltaEOG_v.reshape((-1,1)))

                            history_dY.append(y_pred)
                            
                            if len(history_dY) > 1:
                                deltaDeltaY = history_dY[-1] - history_dY[-2]
                                print(deltaDeltaY, deltaEOG_v, slope)
                                 # only updkcate history_rt with predictions if there is no blink or its been 10 samples since a blink
                                update_cursor(-deltaDeltaY)
                                
                        else: 
                            #Some event action here
                            mouse.click((Button.left))
                            print('blink')
                            after_blink = 20
                            

                        
 # pause predictions being appended to history_rt for 10 samples after a blink is detected

                readings = 0
                plt.pause(0.01)
            
        except ValueError:
            pass
        except UnicodeDecodeError:
            pass
    if close_program:
            #print(len(y_data_arr))
            with open("rt.json", "w") as f:
                    json.dump(rt.tolist(), f)

            # with open("history_rt.json", "w") as f:
            #         json.dump(history_rt, f)
            break
    

    
ser.close()
# plt.ioff()  # Turn off interactive mode
# plt.show()  # Keep the plot open after program finisheskc