import tkinter as tk
import numpy as np
from PIL import Image, ImageGrab

class DigitGUI:

    def __init__(self, model):
        self.model = model
        
        self.root = tk.Tk()
        self.root.title("MNIST Digit Recognizer")

        self.canvas = tk.Canvas(self.root, width=200, height=200, bg="white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        tk.Button(self.root, text="Predict", command=self.predict_digit).pack()
        self.label = tk.Label(self.root, text="Draw a digit", font=("Arial", 16))
        self.label.pack()

        def clear_canvas():
            self.canvas.delete("all")

        clear_button = tk.Button(self.root, text="Clear", command=clear_canvas)
        clear_button.pack()

        self.last_x, self.last_y = None, None

    def draw(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=15)
        self.last_x, self.last_y = event.x, event.y

    def preprocess(self, img):
        img = img.convert("L")

        # binarize
        img = img.point(lambda x: 0 if x < 200 else 255)
        arr = np.array(img)

        # find digit bbox
        coords = np.where(arr < 200)
        if len(coords[0]) == 0:
            return np.zeros((1, 28, 28))

        ymin, ymax = coords[0].min(), coords[0].max()
        xmin, xmax = coords[1].min(), coords[1].max()
        arr = arr[ymin:ymax, xmin:xmax]

        h, w = arr.shape

        # resize to 20
        if h > w:
            new_w = int(20 * w / h)
            arr = Image.fromarray(arr).resize((new_w, 20))
        else:
            new_h = int(20 * h / w)
            arr = Image.fromarray(arr).resize((20, new_h))

        arr = np.array(arr)

        # pad to 28x28
        h, w = arr.shape
        padded = np.zeros((28, 28))
        x_offset = (28 - h) // 2
        y_offset = (28 - w) // 2
        padded[x_offset:x_offset+h, y_offset:y_offset+w] = arr

        return padded.astype("float32") / 255.0
    
    def predict_digit(self):
        def predict(network, images):
            for layer in network:
                images = layer.forward(images)
            return images

        # Get canvas region
        x1 = self.root.winfo_rootx() + self.canvas.winfo_x()
        y1 = self.root.winfo_rooty() + self.canvas.winfo_y()
        x2 = x1 + self.canvas.winfo_width()
        y2 = y1 + self.canvas.winfo_height()

        img = ImageGrab.grab().crop((x1, y1, x2, y2))

        inp = self.preprocess(img).reshape(1, 28, 28, 1)

        pred = predict(self.model, inp).argmax()

        self.label.config(text=f"Prediction: {pred}")
        self.last_x, self.last_y = None, None

    def run(self):
        self.root.mainloop()