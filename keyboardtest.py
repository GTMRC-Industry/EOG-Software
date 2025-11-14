from pynput import keyboard

def on_press(key):
    try:
        if key.char == 'k':
            k = True
            print(k)
        elif key.char == 'b':
            print('berfaersvd')
    except AttributeError:
        # handle special keys (e.g. space, enter)
        pass
listener = keyboard.Listener(on_press=on_press)
listener.start()

print("Listening for 'k' — press it to test.")
listener.join()




