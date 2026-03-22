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
serial_port = '/dev/cu.usbmodem11301'  # Replace with your Arduino's serial port (e.g., '/dev/cu.usbmodem101' on Linux or 'COM3' on Windows)
baud_rate = 230400
ser = serial.Serial(serial_port, baud_rate)
history = 500
update_batch_size = 20
update_cursor_batch_size = 0
a = 0.05  # Exponential Smoothing Parameter
baseline = 1023 / 2

# Plot Setup
plt.ion()
fig, ax = plt.subplots()
x_data = deque(maxlen=history)
y_data = deque(maxlen=history)
y_pred_deque = deque(maxlen=history)
line, = ax.plot([], [], 'r-')
ax.set_xlim(0, history)
ax.set_ylim(0, 1000)

# Initialize data stream lists
y_data_list = []
y_data_arr = []
blink_data_list = []
blink_data_arr = []

# Initialize 'history' lists to keep track of predicted values in real-time 
history_dY = []
history_dEOG = []
history_slope = []

# Initialize cooldown variable used later in real-time prediction
cooldown = 0

#Feature Extraction Setup
global open_blink_calibration
global record_blink
global open_eye_mvmt_calibration
global record_eye_mvmt
global close_program
global d_count


# Initialize boolean flags for keyboard listener
open_blink_calibration = False
record_blink = False
open_eye_mvmt_calibration = False
record_eye_mvmt = False
close_program = False

# Initialize boolean flags for calibration handling
calibrated = False
calibrating = False 
press=False
p = None

blink_calibrating = False
blink_calibrated = False
blink_press = False
n = None


# Initialize counter to only take readings after the first point during saccade calibration 
d_count = 0


print("STARTING")

# Initialize Keyboard Listener
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

        elif key.char == 'c':
            open_eye_mvmt_calibration = True

        elif key.char == 'd':
            d_count += 1
            record_eye_mvmt = True

        elif key.char == 'q':
            close_program = True

    except AttributeError:
        # handle special keys (e.g. space, enter)
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()


# Feature extraction for training and fitting the linear regression model 
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
    window_size = 75
    for entry in y_data_arr:

        entry = savgol_filter(entry, 31, 3)   
        entry = entry - np.mean(entry[:10])

        i = 0
        window_deltas_y = []
        
        # Characterize whether the saccade is positive or negative and then perform sliding window to get the largest deltaV (characteristic deltaV)
        if max(entry, key=abs) > entry[0]:

            argmax = np.argmax(entry)
            entry = entry[:argmax]

            for i in range(len(entry) - window_size):
                window_deltas_y.append(max(entry[i: i + window_size]) - min(entry[i:i+ window_size]))
        else:

            argmin = np.argmin(entry)
            entry = entry[:argmin]

            for i in range(len(entry) - window_size):
                window_deltas_y.append(min(entry[i: i + window_size]) - max(entry[i:i+ window_size]))

        deltaEOG_v = np.append(deltaEOG_v, max(window_deltas_y, key=abs))

    return deltaEOG_v

# Filter outliers out before fitting
def filter_data(deltaEOG_v, deltaY):
    if len(deltaEOG_v)==len(deltaY):
        model_v = RANSACRegressor(residual_threshold=5.0)
        model_v.fit(deltaEOG_v.reshape(-1, 1), deltaY)
        inlier_mask = model_v.inlier_mask_
        deltaEOG_v = deltaEOG_v[inlier_mask]
        deltaY = deltaY[inlier_mask]

        return deltaEOG_v, deltaY
    else:
        print("deltaEOG_v len:", len(deltaEOG_v), "std:", np.std(deltaEOG_v))
        print("deltaY len:", len(deltaY), "std:", np.std(deltaY))
        sys.stdout.flush()


# Fit the linear regression model and save the slope and intercept
def train_model(deltaEOG_v, deltaY):
    model = LinearRegression()
    model.fit(deltaEOG_v.reshape(-1, 1), deltaY)
    # Save the trained model coefficients for later use
    with open("model_coefficients.json", "w") as f:
        json.dump({"slope": model.coef_[0], "intercept": model.intercept_}, f)
    print("Model trained and coefficients saved.")
    return model


# Set deltaV and slope thresholds for blinks
def set_blink_threshold(blink_data):

    deltaEOG_blinks = np.array([])
    # slopes = np.array([])
    # accels = np.array([])

    # Use sliding window to get the max deltaEOG, then return the min of those max deltaEOG's as our threshold.
    window_size = 100
    for entry in blink_data:

        entry = savgol_filter(entry, 31, 3)

        argmax = np.argmax(entry)
        entry = entry[:argmax]

        i = 0
        window_deltas_y = []
        
        for i in range(len(entry) - window_size):
            window_deltas_y.append(max(entry[i: i + window_size]) - min(entry[i:i+ window_size]))

        max_diff_loc = np.argmax(window_deltas_y)
        max_window_bounds = (max_diff_loc, max_diff_loc + window_size)

        deltaEOG_blinks = np.append(deltaEOG_blinks, max(window_deltas_y, key=abs))
        # slopes = np.append(slopes, max(v[max_window_bounds[0] : max_window_bounds[1]]))
        # accels = np.append(accels, max(a[max_window_bounds[0] : max_window_bounds[1]]))
   
    # deltaEOG_blink_std = np.std(deltaEOG_blinks)
    # slope_std = np.std(slopes)
    deltaEOG_blink_lowest = np.mean(deltaEOG_blinks)
    with open ("blink_thresh.json", "w") as f:
        json.dump(deltaEOG_blink_lowest, f)

    return deltaEOG_blink_lowest


def classify(deltaEOG_v, deltaEOG_blink_lowest):
    if deltaEOG_v < deltaEOG_blink_lowest:
        blink = False
    else:
        blink = True

    return blink

# MAIN LOOP
readings = 0
while True:
    if ser.in_waiting > 0:
        try:
            data = ser.readline().decode('utf-8').strip().split(',')[0]  # Read and decode data from serial
            value = int(data)
            if y_data:
                value = value * a + (1 - a) * y_data[-1] # ADD SMOOTHING ALGORITHMS HERE
            x_data.append(len(x_data))
            y_data.append(value)

            # Blink calibration phase
            if open_blink_calibration and not blink_calibrating:

                print('Opening Blink Game')
                n = subprocess.Popen(["python3", "./Calibration_Games/blink_game.py"])
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

                # If calibration finished...
                if n.poll() is not None: 
                    # plt.close(fig)
                    print('Blink Calibration Complete, Setting Blink Thresholds')

                    with open('raw_blinks.json', "w") as f:
                        json.dump(blink_data_arr, f)

                    deltaEOG_blink_lowest = set_blink_threshold(blink_data_arr)
                    print('blink deltaEOG threshold' , deltaEOG_blink_lowest)
                    blink_calibrating = False
                    blink_calibrated = True
                    print(blink_calibrating)
            


            # Saccade calibration phase
            if open_eye_mvmt_calibration and not calibrating: # press c to open the calibration game
                print('Opening Calibration Game')
                p = subprocess.Popen(["python3", "./Calibration_Games/eye_mvmt_calibration_game.py"]) #run the calibration game
                calibrating = True
            
            if calibrating and d_count > 0:
                y_data_list.append(value) # while the game is being run...
                if record_eye_mvmt: # collect calibration data
                    if (not press):
                        press=True
                        record_eye_mvmt = False
                        current_idx = len(y_data_list) - 1
                        print('Captured Eye Movement')


                        y_data_arr.append(y_data_list[:current_idx])
                        y_data_list = []
                else:
                    press=False

                
                # If saccade calibration finished...
                if p.poll() is not None:
                    
                    # Load the points generated in the calibration game
                    with open("calibration_points.json") as f: 
                        points = json.load(f)
                    
                    # Remove the first recorded set of data (random movements)
                    y_data_arr.pop(0)

                    with open("raw_eye_movements.json", "w") as f:
                            json.dump(y_data_arr, f)


                    deltaEOG_v, deltaY = get_deltaEOG_train(y_data_arr), get_deltaY(points)

                    # Dump json files of the recorded eye movements and points for future investigation
                    with open("deltaEOG_v_eye_movements.json", "w") as f:
                        json.dump(deltaEOG_v.tolist(), f)

                    with open("deltaY.json", "w") as f:
                        json.dump(deltaY.tolist(), f)
                    


                    #This filters out any blinks that may have occurred during calibration
                    filtered_deltaEOG_v = deltaEOG_v[[not classify(entry, deltaEOG_blink_lowest) for entry in deltaEOG_v]] # filter out blinks that occurred in the eye movement calibration
                    deltaY = deltaY[[not classify(entry, deltaEOG_blink_lowest) for entry in deltaEOG_v]] # align points with the newly filtered out eye movements
                    print('new length of deltaY: ', len(deltaY))


                    # deltaEOG_v, deltaY = filter_data(deltaEOG_v, deltaY)
                    # print("Filtering data")
                    print('before training')
                    model=train_model(filtered_deltaEOG_v, deltaY)

                    print('after training')
                    update_batch_size = 40

                    # Stop the loop
                    calibrating = False
                    calibrated = True
                    close_program = True

            readings += 1

        except ValueError:
            pass
        except UnicodeDecodeError:
            pass

    if close_program:
        break

ser.close()
