import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

# Doing this to import module from parent folder
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_predictions


# AI code to draw digits and get predictions
class DigitDrawer:
    def __init__(self, master):
        self.master = master
        self.last_x, self.last_y = None, None

        self.canvas = tk.Canvas(master, width = 560, height = 560, bg='black')
        self.canvas.pack()

        self.image = Image.new('L', (560, 560), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)

        self.clear_button = tk.Button(master, text = 'Clear', command = self.clear_canvas)
        self.clear_button.pack()

        self.prediction_label = tk.Label(master, text = 'Prediction: None', font = ('Arial', 16))
        self.prediction_label.pack()


    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw_line(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill = 'white', width = 75, capstyle = tk.ROUND, smooth = tk.TRUE)
            
            self.draw.line([self.last_x, self.last_y, event.x, event.y], fill = 255, width = 75, joint = 'curve')
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
            self.prediction_label.config(text = 'Prediction: None')
            return

        predicted_values = make_predictions.get(img_array)
        digit = np.argmax(predicted_values, axis = 0)[0]
        confidence = predicted_values[digit][0] * 100

        self.prediction_label.config(text = f'Prediction: {digit} ({confidence:.1f}%)')


    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (560, 560), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text = 'Prediction: None')


root = tk.Tk()
app = DigitDrawer(root)
root.mainloop()
