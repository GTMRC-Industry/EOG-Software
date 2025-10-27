import tkinter as tk
import random 

root = tk.Tk()
root.attributes('-fullscreen', True)
root.attributes('-topmost', True)
root.update_idletasks()

canvas = tk.Canvas(root, bg='white')
canvas.pack(fill='both', expand=True)


root.update()

#Coordinates 
x = 0
y = 0
height = root.winfo_height()
width = root.winfo_width()
r = 10  #Radius, change if needed
n = 50
circle_count = 0
points=[]
def on_spacebar(event):
    global circle_count
    if circle_count < n:
        canvas.delete("all")  #Delete
        rx = random.randint(x + r, width - r)
        ry = random.randint(y + r, height - r)
        points.append((rx,ry))
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
        try:
            canvas.create_line(rx, ry, px, py, arrow=tk.FIRST, width=2, fill='red')
        except:
            pass
        circle_count += 1
        if circle_count >= n:
            print(f"Maximum of {n} circles reached!")

#Binds spacebar
root.bind('<space>', on_spacebar)

#TO use events
root.focus_set()

#Escape fullscreen
root.bind('<Escape>', lambda e: root.attributes('-fullscreen', False))

root.mainloop()