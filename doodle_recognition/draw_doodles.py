
# HD fix
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

# Doing this to import module from parent folder
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_predictions


# AI code to draw doodles and get predictions
classes = ['Apple', 'Car', 'Crown', 'Flower', 'Fork', 'Hand', 'Headphones', 'Keyboard', 'Light Bulb', 'Moustache', 'Radio', 
           'Rifle', 'Stairs', 'Stop Sign', 'Tree']

class DigitDrawer:
    def __init__(self, master):
        self.master = master
        self.last_x, self.last_y = None, None

        self.main_frame = tk.Frame(master)
        self.main_frame.pack()

        self.canvas = tk.Canvas(self.main_frame, width = 560, height = 560, bg = 'black')
        self.canvas.grid(row = 0, column = 0)

        self.image = Image.new('L', (560, 560), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)

        self.clear_button = tk.Button(master, text = 'Clear', command = self.clear_canvas)
        self.clear_button.pack()

        self.pred_frame = tk.Frame(self.main_frame)
        self.pred_frame.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = 'n')

        self.pred_labels = []
        for cls in classes:
            lbl = tk.Label(self.pred_frame, text = f'{cls}: 0.0%', anchor = 'w', font = ('Arial', 12))
            lbl.pack(fill = 'x')
            self.pred_labels.append(lbl)

        self.best_label = tk.Label(master, text = 'Prediction: None', font = ('Arial', 14, 'bold'))
        self.best_label.pack(pady = 10)


    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw_line(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill = 'white', width = 15, capstyle = tk.ROUND, smooth = tk.TRUE)
            self.draw.line([self.last_x, self.last_y, event.x, event.y], fill = 255, width = 15, joint = 'curve')
            self.last_x, self.last_y = event.x, event.y
            self.predict_current_digit()

    def stop_draw(self, event):
        self.last_x, self.last_y = None, None


    def preprocess_image(self):
        img_array = np.array(self.image)

        coords = np.argwhere(img_array > 0)
        if coords.size == 0:
            return None
        
        y0, x0 = coords.min(axis = 0)
        y1, x1 = coords.max(axis = 0) + 1
        digit = img_array[y0:y1, x0:x1]
        digit_img = Image.fromarray(digit)

        max_side = max(digit_img.size)
        scale = 20.0 / max_side
        new_size = (max(1, int(digit_img.size[0] * scale)), max(1, int(digit_img.size[1] * scale)))
        digit_img = digit_img.resize(new_size, Image.LANCZOS)

        centered_img = Image.new('L', (28, 28), 0)
        paste_x = (28 - digit_img.size[0]) // 2
        paste_y = (28 - digit_img.size[1]) // 2
        centered_img.paste(digit_img, (paste_x, paste_y))

        img_array = np.array(centered_img).astype(np.float32) / 255.0
        return img_array.reshape(784, 1)


    def predict_current_digit(self):
        img_array = self.preprocess_image()

        if img_array is None:
            for lbl in self.pred_labels:
                lbl.config(text = 'No drawing')

            self.best_label.config(text = 'Prediction: None')
            return

        predicted_values = make_predictions.get(img_array, 'doodle_recognition') 
        for i, prob in enumerate(predicted_values[:, 0]):
            self.pred_labels[i].config(text = f'{classes[i]}: {prob * 100 :.1f}%')

        best_idx = np.argmax(predicted_values[:, 0])
        best_prob = predicted_values[best_idx][0] * 100
        self.best_label.config(text = f'Prediction: {classes[best_idx]} ({best_prob :.1f}%)')


    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (560, 560), 0)
        self.draw = ImageDraw.Draw(self.image)

        for lbl, cls in zip(self.pred_labels, classes):
            lbl.config(text = f'{cls}: 0.0%')

        self.best_label.config(text = 'Prediction: None')


root = tk.Tk()
root.resizable(False, False)
app = DigitDrawer(root)
root.mainloop()
