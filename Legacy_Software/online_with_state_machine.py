#NOTES FOR SPRING 2026: SOFTWARE WORKS, JUST NEED VERY CLEAN SIGNAL, CALIBRATION SLOPES ARE INCONSISTENT, ADD HORIZONTAL AXIS. 




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
serial_port = '/dev/cu.usbmodem11101'  # Replace with your Arduino's serial port (e.g., '/dev/cu.usbmodem101' on Linux or 'COM3' on Windows)
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

rt = [] #define real time list that will populate with data


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
history_last_val = []

# Initialize cooldown variable used later in real-time prediction
cooldown = 0


#STATE MACHINE VARIABLES
upper_thresh = 0
lower_thresh = 0
baseline_thresh = 0

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
    with open("linear_model_coefficients.json", "w") as f:
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
        # v = savgol_filter(entry, 31, 3, deriv=1)
        # a = savgol_filter(entry, 31, 3, deriv=2)

        argmax = np.argmax(entry)
        entry = entry[:argmax]
        # v = v[:argmax]
        # a = a[:argmax]

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
    # slope_lowest = np.min(slopes) - slope_std
    # accel_lowest = np.mean(accels)

    return deltaEOG_blink_lowest



# Blink vs. Saccade classification based on changes in slope or deltaV thresholds
def classify(deltaEOG_v, deltaEOG_blink_lowest, deltaSlope):
    if deltaSlope > 3 or deltaEOG_v > deltaEOG_blink_lowest:
        blink = True

    else:
        blink = False

    return blink

# Classifier for training to filter out blinks during saccade calibration
def classify_only_peak_threshold(deltaEOG_v, deltaEOG_blink_lowest):
    if deltaEOG_v < deltaEOG_blink_lowest:
        blink = False
    else:
        blink = True

    return blink

# Extract features on real-time data
def get_deltaEOG_test(y_data):
    deltaEOG_v = np.array([])
    # Sliding windowkc
    window_size = 75
    i = 0
    window_deltas_y = []
    entry=list(y_data)
    entry = entry - np.mean(entry[:10]) # baseline normalization


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

        argmin = np.argmin(entry)
        entry = entry[:argmin]
        for i in range(len(entry) - window_size):
            # signal went up first, then down → negative saccade
            window_deltas_y.append(min(entry[i: i + window_size]) - max(entry[i:i+ window_size]))
        slope = 0
        accel = 0

    deltaEOG_v = np.append(deltaEOG_v, max(window_deltas_y, key=abs))

    return deltaEOG_v, slope, accel

# Update the plot in real-time
def update_plot():
    line.set_xdata(np.arange(len(x_data)))  # X-axis
    line.set_ydata(np.array(y_data))  # Y-axis
    plt.draw()

# Update a different plot that plots the users eye movements in real-time
def update_point_plot():
    line.set_xdata(np.arange(len(x_data)))
    line.set_ydata(np.array(y_pred_deque))
    plt.draw()




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

    with open('linear_model_coefficients.json') as f:
        model_coef = json.load(f)

    SCALE = 1

    scaled_delta = deltaY * SCALE

    # Set the max and min bound from the edges of the screen
    bound = 20 

    current_y = mouse.position[1]

    # Handling if the cursor's new position would have exceeded screen boundaries
    if current_y + scaled_delta > screen_h:
        mouse.move(0, screen_h - current_y - bound)
    elif current_y + scaled_delta < 0:
        mouse.move(0, 0 - current_y + bound)
    else:
        mouse.move(0, scaled_delta)









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
            rt.append(value)

            # Blink calibration phase
            if open_blink_calibration and not blink_calibrating:

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

                # If calibration finished...
                if n.poll() is not None: 
                    # plt.close(fig)
                    print('Blink Calibration Complete, Setting Blink Thresholds')

                    with open('raw_blinks.json', "w") as f:
                        json.dump(blink_data_arr, f)

                    deltaEOG_blink_lowest = set_blink_threshold(blink_data_arr)
                    print('blink deltaEOG threshold' , deltaEOG_blink_lowest)
                    # print('blink slope threshold' , slope_lowest)
                    # print('blink accel threshold' , accel_lowest)
                    blink_calibrating = False
                    blink_calibrated = True
                    print(blink_calibrating)
            


            # Saccade calibration phase
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
                    filtered_deltaEOG_v = deltaEOG_v[[not classify_only_peak_threshold(entry, deltaEOG_blink_lowest) for entry in deltaEOG_v]] # filter out blinks that occurred in the eye movement calibration
                    deltaY = deltaY[[not classify_only_peak_threshold(entry, deltaEOG_blink_lowest) for entry in deltaEOG_v]] # align points with the newly filtered out eye movements
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

            readings += 1

            # When both blink and saccade calibration are finished, start predicting in real-time
            if (readings > update_batch_size) and (not calibrating) and (not blink_calibrating):

                # Update the plot 
                # update_plot()
                if calibrated and blink_calibrated: # if calibration finished

                                                    # # Perform feature extraction on incoming real-time data
                                                    # deltaEOG_v, slope, accel = get_deltaEOG_test(y_data)

                    ### ESTABLISH BASELINE
                    #check if there is a flat trend
                    if (abs(rt[-1] - rt[-2]) < baseline_thresh and abs(rt[-2] - rt[-3]) < baseline_thresh):
                        baseline = np.mean(rt[-1], rt[-2], rt[-3])

                    


                    #see if that value between windows is flat






                        # Keep track of the incoming deltaVs and their respective slopes
                    history_dEOG.append(deltaEOG_v)
                    history_slope.append(slope)

                    if len(history_slope) > 1:

                        # Calculate the change in slope between windows, use for blink vs. saccade classification
                        deltaSlope = history_slope[-1] - history_slope[-2]
                        classification = classify(deltaEOG_v, deltaEOG_blink_lowest, deltaSlope)
                        
                    else:
                        classification = classify(deltaEOG_v, deltaEOG_blink_lowest, 0)
                        
                    # If saccade......
                    if not classification:

                        # Adjust this noise threshold below depending on how noisy the signal is. Noise can generate some deltaV which we don't want to account for. 
                        if abs(deltaEOG_v) < 35:
                                deltaY = 0
                        else:
                            deltaY = model.predict(deltaEOG_v.reshape((-1,1)))

                        # Keep track of deltaY_predictions. 
                        history_dY.append(deltaY)
                        
                        
                        if len(history_dEOG) > 1:
                            
                            # Identify where the predictions flip signs. When this occurs we want to stop predicting for a couple samples to make sure eye movements won't rebound. 
                            if (history_dEOG[-1] * history_dEOG[-2]) < 0 and history_dEOG[-1] != 0:
                                print('transition')
                                history_dY[-1] = 0
                                deltaDeltaY = 0
                                tracking = False
                                refract = True
                            else:

                                # Calculate the change in predicted deltaY values to get the effective amount the cursor should move (remember that multiple windows of one eye movement can predict the same deltaY, but we only want to move once per eye movement)
                                deltaDeltaY = history_dY[-1] - history_dY[-2]

                            if deltaDeltaY != 0:
                                print(deltaDeltaY, deltaEOG_v, slope)
                            
                            # Update the cursor with the effective delta (use negative sign because for the Y-axis, 0 is the top and max is at the bottom)
                            update_cursor(-deltaDeltaY)

                    # If blink......
                    if classification:
                        
                        # Left-click the mouse
                        mouse.click((Button.left))
                        print('blink')

                        # Start a cooldown on predictions
                

                readings = 0
                plt.pause(0.01)
            
        except ValueError:
            pass
        except UnicodeDecodeError:
            pass
    if close_program:

            break
    

    
ser.close()
# plt.ioff()  # Turn off interactive mode
# plt.show()  # Keep the plot open after program finisheskc