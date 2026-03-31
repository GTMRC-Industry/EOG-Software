import sys
import serial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

update_batch_size = 20
update_cursor_batch_size = 0
a = 0.05  # Exponential Smoothing Parameter
baseline = 1023 / 2

# Plot Setup
plt.ion()
fig, ax = plt.subplots()
y_data = []
line, = ax.plot([], [], 'r-')

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


with open("./model_coefficients.json") as f: # load the list of random points generated in calibration_game
    model_coefficients = json.load(f)

model_slope = model_coefficients['slope']
model_intercept = model_coefficients['intercept']


with open("./blink_thresh.json") as f:
    blink_thresh = json.load(f)

def classify(deltaEOG_v, blink_thresh):
    if deltaEOG_v < blink_thresh:
        blink = False
    else:
        blink = True

    return blink

###NEW ALGORITHM PARAMETERS###


baseline = 0
threshold = 50
noise_threshold = 10
lower_bound = 0
upper_bound = 0
deltaEOG_v = 0

mov_avg_pt = 10 #use a 10 point moving average (for noise handling)
hist_mov_avg = [] #keep track of the moving average values so we can compare present vs past values




hit_max = False
hit_min = False




s1 = pd.read_csv('./recordings/csv_outputs/UDB1.csv') # s1 = pd.read_csv('./measurements/csv/s1.csv')
timestep = np.mean(np.diff(s1['timestamp'])) # time

def reconstruct(sim, timestep):
    global baseline
    global threshold
    global upper_bound
    global lower_bound
    global noise_threshold
    global deltaEOG_v


    for i in range(sim.shape[0]):
        y_data.append(sim['data1'][i]) # signal
        time.sleep(timestep)
        if len(y_data) > 0 and len(y_data) % mov_avg_pt == 0: #Note that by using MOD there will be no overlap between the groups of averages. 
            mov_avg = np.mean(y_data[-mov_avg_pt : -1])

            hist_mov_avg.append(mov_avg)

            if len(hist_mov_avg) > 1:

                if abs(hist_mov_avg[-1] - hist_mov_avg[-2]) < noise_threshold:
                    print(abs(hist_mov_avg[-1] - hist_mov_avg[-2]))
                    baseline = hist_mov_avg[-1]
                    lower_bound = baseline - threshold
                    upper_bound = baseline + threshold

                    hit_max = False
                    hit_min = False

                    # print(baseline, lower_bound, upper_bound)

                elif hist_mov_avg[-1] > upper_bound and hist_mov_avg[-1] < hist_mov_avg[-2] and not hit_max:
                    hit_max = True
                    max_y = hist_mov_avg[-2]
                    deltaEOG_v = max_y - baseline
                    print(deltaEOG_v)

                elif hist_mov_avg[-1] < lower_bound and hist_mov_avg[-1] > hist_mov_avg[-2] and not hit_min:
                    hit_min = True
                    min_y = hist_mov_avg[-2]
                    deltaEOG_v = min_y - baseline
                    print(deltaEOG_v)

                
                classification = classify(deltaEOG_v, blink_thresh)

                if classification: 
                    mouse.click((Button.left))
                    print('blink')

                elif not classification:
                    deltaY = (model_slope * deltaEOG_v)
                    update_cursor(deltaY)

    

reconstruct(s1, timestep)