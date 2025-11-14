import tkinter as tk

# Initial countdown value
count = 10
rep = 0 



def on_key_press(event):
    global count
    if event.char.lower() == 'b':  # check if the key pressed is 'b'
        if count > 0:
            count_down()
        else:
            canvas.itemconfig(text_item, text="Done!")
            root.destroy()

def count_down():
    global count
    canvas.itemconfig(text_item, text=str(count))
    count -= 1

# Create main window
root = tk.Tk()
root.title("Countdown on Canvas")
root.attributes('-fullscreen', True)
root.attributes('-topmost', True)
root.update()
height = root.winfo_height()
width = root.winfo_width()
# Create canvas

canvas = tk.Canvas(root, bg="black")
canvas.pack(fill="both",expand=True)
height = root.winfo_height()
width = root.winfo_width()
# Add text to canvas
text_item = canvas.create_text((width / 2), (height / 2), text=str(count), fill="white", font=("Helvetica", 48))

canvas.itemconfig(text_item, text = 'Blink after every time you press B, press B to start')
def close_window(event):
    root.destroy()
# Bind key press event
root.bind("<KeyPress>", on_key_press)
root.bind('q',close_window)
root.protocol("WM_DELETE_WINDOW", close_window)
canvas.pack(fill='both', expand=True)
root.mainloop()
