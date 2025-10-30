import tkinter as tk
import random 
import json
import os
 

root = tk.Tk()
root.attributes('-fullscreen', True)
root.attributes('-topmost', True)
root.update_idletasks()

canvas = tk.Canvas(root, bg='white')
canvas.pack(fill='both', expand=True)


root.update()

# Make sure Tk actually captures keyboard events
root.focus_force()
root.lift()
root.after(100, lambda: root.focus_force())

#Coordinates 
x = 0
y = 0
height = root.winfo_height()
width = root.winfo_width()
r = 10  #Radius, change if needed
n = 50
circle_count = 0
points=[]

#Draw Initial Center circle so that user knows where to start
canvas.create_oval((width/2 - r), (height/2 - r), (width/2 + r), (height/2 + r), fill='blue', outline='black')

points_converted=[]
#Function to translate coordinates
def convert(x,y,w=width,h=height):
    #(width/2,height/2) is center!
    x1=x-w/2
    y1=-(y-h/2)
    return (x1,y1)

def draw_random_oval(event):
    global circle_count
    if circle_count < n:
        canvas.delete("all")  #Delete
        rx = random.randint(x + r, width - r)
        ry = random.randint(y + r, height - r)
        coordinates_conv=convert(rx,ry)
        points.append((rx,ry))
        points_converted.append(coordinates_conv)
        #Immeadiate previous one on disc lines
        try:
            px=points[circle_count-1][0] #Previous
            py=points[circle_count-1][1]
            canvas.create_oval(px - r, py - r, px + r, py + r,
                outline='black',
                width=2,
                fill='',
                dash=(5, 5))
        except:
            pass
        canvas.create_oval(rx - r, ry - r, rx + r, ry + r, fill='blue', outline='black')
        if circle_count == 0: 
            try:
                canvas.create_oval((width/2) - r, (height/2) - r, (width/2)  + r, (height/2) + r,
                outline='black',
                width=2,
                fill='',
                dash=(5, 5))
                canvas.create_line(rx, ry, (width/2), (height/2), arrow=tk.FIRST, width=2, fill='red')
            except:
                pass
        else: 
            try:
                canvas.create_line(rx, ry, px, py, arrow=tk.FIRST, width=2, fill='red')
            except:
                pass
        circle_count += 1
        if circle_count >= n:
            print(f"Maximum of {n} circles reached!")
            print(points)
            root.destroy()
#Binds spacebar
root.bind('<space>', draw_random_oval)

#TO use events
root.focus_set()

#Escape fullscreen
root.bind('<Escape>', lambda e: root.attributes('-fullscreen', False))


root.mainloop()

with open("calibration_points.json", "w") as f:
    json.dump(points_converted, f)