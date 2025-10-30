from pynput import keyboard

def on_press(key):
    try:
        if key.char == 'k':
            print("k pressed!")
    except AttributeError:
        # handle special keys (e.g. space, enter)
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

print("Listening for 'k' — press it to test.")
listener.join()






