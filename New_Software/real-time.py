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
y_data = []
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
global close_program


# Initialize boolean flags for keyboard listener
close_program = False

def on_press(key):
    global close_program
    try:
        close_program = False

        if key.char == 'q':
            close_program = True

    except AttributeError:
        # handle special keys (e.g. space, enter)
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

# Setup cursor movements around the screen and clicks 
import pyautogui
from pynput.mouse import Controller, Button
mouse = Controller()
import time

screen_w, screen_h = pyautogui.size()

cursor_x = screen_w // 2
cursor_y = screen_h // 2

print(cursor_x, cursor_y)

# Start the cursor at the center of the screen
mouse.position = (cursor_x, cursor_y)

# Update the cursors position given a change in Y
def update_cursor(deltaY):

    # Set the max and min bound from the edges of the screen
    bound = 20 

    current_y = mouse.position[1]

    # Handling if the cursor's new position would have exceeded screen boundaries
    if current_y + deltaY > screen_h:
        mouse.move(0, screen_h - current_y - bound)
    elif current_y + deltaY < 0:
        mouse.move(0, 0 - current_y + bound)
    else:
        mouse.move(0, deltaY)

# Initialize boolean flags for calibration handling


with open("model_coefficients.json") as f: # load the list of random points generated in calibration_game
    model_coefficients = json.load(f)

model_slope = model_coefficients['slope']
model_intercept = model_coefficients['intercept']


with open("blink_thresh.json") as f:
    blink_thresh = json.load(f)

def classify(deltaEOG_v, blink_thresh):
    if deltaEOG_v < blink_thresh:
        blink = False
    else:
        blink = True

    return blink

###NEW ALGORITHM PARAMETERS###
baseline = 0
threshold = 100
noise_threshold = 30
lower_bound = 0
upper_bound = 0

lookback = -9



hit_max = False
hit_min = False

while True: 

    if ser.in_waiting > 0:
        try:
            data = ser.readline().decode('utf-8').strip().split(',')[0]  # Read and decode data from serial
            value = int(data)
            if y_data:
                value = value * a + (1 - a) * y_data[-1] # ADD SMOOTHING ALGORITHMS HERE
            y_data.append(value)

            if len(y_data) > 2 * abs(lookback) + 1:

                if abs(np.mean(np.diff(y_data[-1 : lookback]))) < noise_threshold:
                    baseline = np.mean(y_data[-1 : lookback])
                    upper_bound = baseline + threshold
                    lower_bound = baseline - threshold
                    hit_max = False
                    hit_min = False
                    print(baseline, upper_bound, lower_bound)
                
                elif y_data[-1] > upper_bound and not hit_max:

                    if np.mean(y_data[-1 : lookback]) < np.mean(y_data[lookback : 2 * lookback]):
                        max_y = np.max(y_data[-1 : lookback])
                        deltaEOG_v = max_y - baseline
                        hit_max = True

                elif y_data[-1] < lower_bound and not hit_min:
    
                    if np.mean(y_data[-1 : lookback]) > np.mean(y_data[lookback : 2 * lookback]):
                        min_y = np.min(y_data[-1 : lookback])
                        deltaEOG_v = min_y - baseline
                        hit_min = True

            

            classification = classify(deltaEOG_v, blink_thresh)

            if classification: 
                mouse.click((Button.left))
                print('blink')

            elif not classification:
                deltaY = (model_slope * deltaEOG_v)
                update_cursor(deltaY)

            

                








        except ValueError:
            pass
        except UnicodeDecodeError:
            pass
    if close_program:

            break
    

    
ser.close()