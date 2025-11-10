import sys
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
def get_deltaY(points):
    y_points = np.array([])
    for point in points: 
        y_points = np.append(y_points, point[1])

    #Apply first differences to y_points
    deltaY = np.diff(y_points)
    return deltaY

def get_deltaEOG(y_data_arr):
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
    global model
    model = LinearRegression()
    model.fit(deltaEOG_v.reshape(-1, 1), deltaY)
    # Save the trained model coefficients for later use
    with open("linear_model_coefficients.json", "w") as f:
        json.dump({"slope": model.coef_[0], "intercept": model.intercept_}, f)
    print("Model trained and coefficients saved.")


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
                    print('Calibration Complete. Training Model...')
                    #TEST THESE FUNCITONS TODAY
                    deltaEOG_v, deltaY = get_deltaEOG(y_data_arr), get_deltaY(points)
                    deltaEOG_v, deltaY = filter_data(deltaEOG_v, deltaY)
                    print("Filtering data")
                    train_model(deltaEOG_v, deltaY)

                    calibrating = False # stop the loop
                    print(calibrating)
            #End Calibration phase
            readings += 1
            if (readings > update_batch_size) and (not calibrating):
                update_plot()
                #Call Model.predict here
                deltaEOG_v = get_deltaEOG(y_data)
                y_pred = model.predict(deltaEOG_v)
                print(y_pred)
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