# Modified calibration_game.py with fade transitions
import tkinter as tk
import random
import json
import os

root = tk.Tk()
root.attributes('-fullscreen', True)
root.attributes('-topmost', True)
root.update_idletasks()

canvas = tk.Canvas(root, bg='black')
canvas.pack(fill='both', expand=True)
root.update()
root.focus_force()
root.lift()
root.after(100, lambda: root.focus_force())

x = 0
y = 0
height = root.winfo_height()
width = root.winfo_width()
r = 20
n = 50
n_per_round=10
n_rounds=n/n_per_round
cooldown=10 #10 seconds for each break
time_left=cooldown
stop=False
circle_count = 0
points = []
points_converted = [(0,0)]

# Fading helpers
# FADE_STEPS = 50

def fade_item(item, start, end, itemtype, mode, callback=None):

    if itemtype == 'point':
        FADE_STEPS = 50
    elif itemtype == 'arrow':
        FADE_STEPS = 20
    else:
        AttributeError
    step = (end - start) / FADE_STEPS

    def fade(i=0):
        if i > FADE_STEPS:
            if callback: callback()
            return
        if mode == 'grayscale':
            color = f"#{int(start + step * i):02x}{int(start + step * i):02x}{int(start + step * i):02x}"
        elif mode == 'red':
            color = f"#{int(start + step * i):02x}0000"
        elif mode == 'yellow':
            color = f"#{int(start + step * i):02x}fe00"
        else: 
            AttributeError
        canvas.itemconfig(item, fill=color)
        root.after(20, lambda: fade(i+1))

    fade()



def convert(x,y,w=width,h=height):
    x1 = x - w/2
    y1 = -(y - h/2)
    return (x1, y1)

# initial center circle
start_circle = canvas.create_oval(width/2 - r, height/2 - r, width/2 + r, height/2 + r, fill='blue', outline='black')

prev_circle_item = start_circle

start_arrow = None

prev_arrow_item = start_arrow
def draw_random_oval(event):
    global circle_count, prev_circle_item, prev_arrow_item, stop
    if (circle_count % n_per_round==0) and (circle_count!=0): #This means its either 10,20,30...
            if stop==True:
                circle_count+=1
            else:
                manage_breaks()
    if stop==False:
        if circle_count >= n:
            root.destroy()
            return

        rx = random.randint(x + r, width - r)
        ry = random.randint(y + r, height - r)

        points.append((rx, ry))
        points_converted.append(convert(rx, ry))

        new_circle = canvas.create_oval(rx - r, ry - r, rx + r, ry + r, fill='white', outline='black')

        # fade out previous, fade in next

        fade_item(prev_circle_item, 255, 0, itemtype='point', mode='grayscale')

        # if (circle_count + 1) % 4 == 0:
        #     fade_item(new_circle, 0, 255, itemtype='point', mode='yellow')
        # else:
        fade_item(new_circle, 0, 255, itemtype='point', mode='grayscale')


        prev_circle_item = new_circle

        # draw arrow
        px, py = points[circle_count - 1] if circle_count > 0 else (width/2, height/2)
        new_arrow = canvas.create_line(px, py, rx, ry, arrow=tk.LAST, width=2, fill='red', dash = (3, 5))

        fade_item(prev_arrow_item, 255, 0, itemtype='arrow', mode='red')
        fade_item(new_arrow, 0, 255, itemtype='arrow', mode='red')

        prev_arrow_item = new_arrow

        circle_count += 1

        if circle_count == n:
            print("Maximum reached")
            print(points_converted)
            root.destroy()


def close_window(event):
    root.destroy()


root.bind('d', draw_random_oval)


root.bind('q', close_window)
root.bind('<Escape>', lambda e: root.attributes('-fullscreen', False))
root.focus_set()

# overlap timing so next point appears early
OVERLAP_MS = 300  # new point appears 300ms before previous fades

# auto-generate a new point every 2 second

def auto_press_d():
    root.event_generate('<KeyPress-d>') # simulate user pressing 'd'
    # if circle_count < n and circle_count % 4 != 0:
    root.after(2000 - OVERLAP_MS, auto_press_d)

    # elif circle_count < n and circle_count % 4 == 0:
    #     print(circle_count, 'blink allowed')
    #     root.after(3000 - OVERLAP_MS, auto_press_d)
def countdown(): #During break, will let the user know time left for next round
    if time_left==0:
        #Erase message

        pass
    pass
    
def reset_round():
    global prev_circle_item,prev_arrow_item, time_left
    prev_circle_item=canvas.create_oval(width/2 - r, height/2 - r, width/2 + r, height/2 + r, fill='blue', outline='black')
    points.append((width/2, height/2))
    points_converted.append(convert(width/2, height/2))
    prev_arrow_item=start_arrow
    #Label for text
    #label = tk.Label(root, text="Time until next round")
    #label.pack()


def manage_breaks():
    global stop, prev_arrow_item, prev_circle_item, time_left
    if stop==False:
        print("LLEGUE")
        stop=True
        fade_item(prev_arrow_item, 255, 0, itemtype='arrow', mode='red')
        fade_item(prev_circle_item, 255, 0, itemtype='point', mode='grayscale')
        root.after(2000-OVERLAP_MS,reset_round)
        time_left=cooldown
        root.after(10000,manage_breaks)
    else:
        stop=False

root.after(2000 - OVERLAP_MS, auto_press_d)

# def auto_step():
#     draw_random_oval()
#     if circle_count < n:
#         root.after(2000 - OVERLAP_MS, auto_step)

# root.after(2000 - OVERLAP_MS, auto_step)

root.mainloop()

with open("calibration_points.json", "w") as f:
    json.dump(points_converted, f)
