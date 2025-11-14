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
import time

# Set up the serial port and parameters
serial_port = '/dev/cu.usbmodem101'  # Replace with your Arduino's serial port (e.g., '/dev/cu.usbmodem101' on Linux or 'COM3' on Windows)
baud_rate = 230400
ser = serial.Serial(serial_port, baud_rate)
history = 500
update_batch_size = 10
a = 0.5 # Exponential Smoothing Parameter
baseline = 1023 / 2

# Plot Setup
plt.ion()
fig, ax = plt.subplots()
x_data = deque(maxlen=history)
y_data = deque(maxlen=history)
line, = ax.plot([], [], 'r-')
ax.set_xlim(0, history)
ax.set_ylim(0, 1023)
y_data_list = []
y_data_arr = []
blink_data_list = []
blink_data_arr = []


#Feature Extraction Setup
global c 
c=0
global open_blink_calibration
global record_blink
global open_eye_mvmt_calibration
global record_eye_mvmt
global close_program

open_blink_calibration = False
record_blink = False
open_eye_mvmt_calibration = False
record_eye_mvmt = False
close_program = False
print("BEFORE LISTENER")
def on_press(key):
    try:
        global open_blink_calibration
        global record_blink
        global open_eye_mvmt_calibration
        global record_eye_mvmt
        global close_program
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
    window_size = 10
    for entry in y_data_arr:
        
        i = 0
        window_deltas_y = []
        
        for i in range(len(entry) - window_size):
            i_max_y, i_min_y = np.argmax(entry), np.argmin(entry)
            if i_max_y > i_min_y:
                # signal went down first, then up → positive saccade
                window_deltas_y.append(max(entry[i: i + window_size]) - min(entry[i:i+ window_size]))
            else:
                # signal went up first, then down → negative saccade
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
    # use sliding window to get the max deltaEOG, then return the min of those max deltaEOG's as our threshold.
    window_size = 25
    for entry in blink_data:
        
        i = 0
        window_deltas_y = []
        
        for i in range(len(entry) - window_size):
            # signal went down first, then up → positive saccade
            window_deltas_y.append(max(entry[i: i + window_size]) - min(entry[i:i+ window_size]))

        deltaEOG_blinks = np.append(deltaEOG_blinks, max(window_deltas_y, key=abs))
    
    deltaEOG_blink_std = np.std(deltaEOG_blinks)
    deltaEOG_blink_lowest = np.min(deltaEOG_blinks) - deltaEOG_blink_std

    return deltaEOG_blink_lowest


# CLASSIFY BLINK OR SACCADE ------------------

def classify(deltaEOG_v, deltaEOG_blink_lowest):
    if deltaEOG_v < deltaEOG_blink_lowest: #if saccade
        blink = False
    else: #if blink
        blink = True

    return blink


# RUN REGRESSION ---------------------
def get_deltaEOG_test(y_data):
    deltaEOG_v = np.array([])
    # Sliding window
    window_size = 10 
    i = 0
    window_deltas_y = []
    entry=list(y_data)
    for i in range(len(entry) - window_size):
        i_max_y, i_min_y = np.argmax(entry), np.argmin(entry)
        if i_max_y > i_min_y:
            # signal went down first, then up → positive saccade
            window_deltas_y.append(max(entry[i: i + window_size]) - min(entry[i:i+ window_size]))
        else:
            # signal went up first, then down → negative saccade
            window_deltas_y.append(min(entry[i: i + window_size]) - max(entry[i:i+ window_size]))

    deltaEOG_v = np.append(deltaEOG_v, max(window_deltas_y, key=abs))

    return deltaEOG_v

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
# Real-time plotting
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
            y_data_list.append(value)
            blink_data_list.append(value)

            # Blink Calibration phase

            if open_blink_calibration and not blink_calibrating: # press k to open the blink game
                print('Opening Blink Game')
                n = subprocess.Popen(["python3", "blink_game.py"])
                blink_calibrating = True
            if blink_calibrating:
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
                    print('Blink Calibration Complete, Setting Blink Threshold')
                    deltaEOG_blink_lowest = set_blink_threshold(blink_data_arr)

                    blink_calibrating = False
                    blink_calibrated = True
                    print(blink_calibrating)

            
            # Eye Movement Calibration phase
            if open_eye_mvmt_calibration and not calibrating: # press c to open the calibration game
                print('Opening Calibration Game')
                p = subprocess.Popen(["python3", "calibration_game.py"]) #run the calibration game
                calibrating = True
            if calibrating: # while the game is being run...
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

                if p.poll() is not None: # when calibration is over
                    with open("calibration_points.json") as f: # load the list of random points generated in calibration_game
                        points = json.load(f)
                    print('Calibration Complete. Training Model...')
                    #TEST THESE FUNCITONS TODAY
                    deltaEOG_v, deltaY = get_deltaEOG_train(y_data_arr), get_deltaY(points)
                    deltaEOG_v, deltaY = filter_data(deltaEOG_v, deltaY)
                    print("Filtering data")
                    model=train_model(deltaEOG_v, deltaY)

                    calibrating = False # stop the loop
                    calibrated = True
                    print(calibrating)
            #End Calibration phase

            readings += 1

            #consider using separate batch sizes to deal with plot lag
            if (readings > update_batch_size) and (not calibrating) and (not blink_calibrating):
                update_plot()
                #Call Model.predict here

                if calibrated and blink_calibrated: # if calibration finished
                    deltaEOG_v = get_deltaEOG_test(y_data)
                    classification = classify(deltaEOG_v, deltaEOG_blink_lowest)
                    if not classification:
                        y_pred = model.predict(deltaEOG_v.reshape((-1,1)))
                        print(y_pred)
                    else: 
                        print('blink')


                readings = 0
                plt.pause(0.01)
            
        except ValueError:
            pass
        except UnicodeDecodeError:
            pass
    if close_program:
            #print(len(y_data_arr))
            for prev_readings in y_data_arr:
                 print(len(prev_readings))
            break
    
ser.close()
plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the plot open after program finishes