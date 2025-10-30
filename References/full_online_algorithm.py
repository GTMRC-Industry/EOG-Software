import serial
import matplotlib.pyplot as plt
import numpy as np
import keyboard
from collections import deque
import tkinter as tk
import random
import subprocess
import json
import os
from sklearn.linear_model import LogisticRegression, LinearRegression, HuberRegressor, RANSACRegressor
import time

# Set up the serial port and parameters
serial_port = 'COM5'  # Replace with your Arduino's serial port (e.g., '/dev/ttyUSB0' on Linux or 'COM3' on Windows)
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
press=False

#Feature Extraction Setup
global c 
c=0  
def feature_extraction(signal):
    signal = np.array(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    deltaV = max_val - min_val
    return deltaV

def process_data(y_data_arr, points):
    global c
    deltaV_list = []
    x_points = []
    y_points = []
    for i, signal in enumerate(y_data_arr):
        deltaV = feature_extraction(signal)
        #print(c)
        deltaV_list.append(deltaV)
    for coord in points:
        x_points.append(coord[0])
        y_points.append(coord[1])
    c+=1 

    return deltaV_list, y_points

def train_linear_reg(dV, coords):
    model = LinearRegression()
    model.fit(dV, coords)

    
calibrated = False
calibrating = False 
p = None


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
            
            # Calibration phase
            if keyboard.is_pressed('c') and not calibrating: # press c to open the calibration game
                print('Opening Calibration Game')
                p = subprocess.Popen(["python3", "calibration_game.py"]) #run the calibration game
                calibrating = True
                calibrated = True
            #print(calibrating)
            if calibrating: # while the game is being run...
                if keyboard.is_pressed(" "): # collect calibration data
                    if (not press):
                        press=True
                        keyboard.release(" ")
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
                    """""
                    deltaV, y = process_data(y_data_arr, points) # feature extraction + linear regression fitting
                    train_linear_reg(deltaV, y)
                    time.sleep(0.5)
                    """
                    calibrating = False # stop the loop 
                    print(calibrating)
            #End Calibration phase
            readings += 1
            if (readings > update_batch_size) and (not calibrating):
                update_plot()
                #Call Model.predict here
                readings = 0
                plt.pause(0.01)
        except ValueError:
            pass
        except UnicodeDecodeError:
            pass
    if keyboard.is_pressed('q'):
            #print(len(y_data_arr))
            for prev_readings in y_data_arr:
                 print(len(prev_readings))
            break
    





ser.close()
plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the plot open after program finishes