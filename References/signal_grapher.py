import serial
import matplotlib.pyplot as plt
import numpy as np
from pynput import keyboard
from collections import deque

# Set up the serial port and parameters
serial_port = '/dev/cu.usbmodem101'  # Replace with your Arduino's serial port (e.g., '/dev/ttyUSB0' on Linux or 'COM3' on Windows)
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
press=False

def update_plot():
    line.set_xdata(np.arange(len(x_data)))  # X-axis
    line.set_ydata(np.array(y_data))  # Y-axis
    plt.draw()
# Real-time plotting
readings = 0


global close_program

close_program = False


def on_press(key):
    try:
        global close_program
        if key.char == 'q':
            close_program = True
    except AttributeError:
        # handle special keys (e.g. space, enter)
        pass
listener = keyboard.Listener(on_press=on_press)
listener.start()
while True:
    if ser.in_waiting > 0:
        try:
            data = ser.readline().decode('utf-8').strip().split(',')[0]  # Read and decode data from serial
            value = int(data)
            # if y_data:
            #     value = value * a + (1 - a) * y_data[-1]
            x_data.append(len(x_data))
            y_data.append(value)
            readings += 1
            if readings > update_batch_size:
                update_plot()
                readings = 0
                plt.pause(0.01)
        except ValueError:
            pass
        except UnicodeDecodeError:
            pass
    if close_program:
        break
            

ser.close()
plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the plot open after program finishes