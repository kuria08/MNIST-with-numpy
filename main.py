import numpy as np
import pandas as pd
import os
import tkinter as tk
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

data = np.array(pd.read_csv('digit-recognizer/train.csv'))

data = np.array(data)
m, n = data.shape
np.random.seed(42)
np.random.shuffle(data)

#data for dev
data_dev = data[0:1000].T
y_dev = data_dev[0]
x_dev = data_dev[1:n] / 255.

#data for train
data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n] / 255.

# labels 0-9 -> (10x1)
def one_hot(label):
    one_hot = np.zeros((10, 1))
    one_hot[label] = 1
    return one_hot

# He initialization
def init():
    # 784 -> 512
    w1 = np.random.randn(512, 784)*np.sqrt(2/784)
    b1 = np.zeros((512, 1))
    # 512 -> 256
    w2 = np.random.randn(256, 512)*np.sqrt(2/512)
    b2 = np.zeros((256, 1))
    # 256 -> 128
    w3 = np.random.randn(128, 256)*np.sqrt(2/256)
    b3 = np.zeros((128, 1))
    # 128 -> 10
    w4 = np.random.randn(10, 128)*np.sqrt(2/128)
    b4 = np.zeros((10, 1))
    return w1, b1, w2, b2, w3, b3, w4, b4

#activation function
def ReLU(x):
    return np.maximum(0, x)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def ReLU_derivative(x):
    return x > 0

def forward_propagetion(w1, b1, w2, b2, w3, b3, w4, b4, x):
    z1 = w1.dot(x) + b1
    x1 = ReLU(z1) 
    z2 = w2.dot(x1) + b2
    x2 = ReLU(z2)
    z3 = w3.dot(x2) + b3
    x3 = ReLU(z3)
    z4 = w4.dot(x3) + b4
    x4 = softmax(z4)
    return z1, x1, z2, x2, z3, x3, z4, x4

#training
def train(x_train, y_train, epochs, learning_rate):
    w1, b1, w2, b2, w3, b3, w4, b4 = init()
    for epoch in range(epochs):
        count_corr = 0
        for i in range(x_train.shape[1]):
            labels = one_hot(y_train[i])
            x = x_train[:,i:i+1]
            z1, x1, z2, x2, z3, x3, z4, x4 = forward_propagetion(w1, b1, w2, b2, w3, b3, w4, b4, x)

            count_corr += int(np.argmax(x4) == np.argmax(labels))

            delta_x4 = x4 - labels
            delta_x3 = w4.T.dot(delta_x4)*ReLU_derivative(z3)
            delta_x2 = w3.T.dot(delta_x3)*ReLU_derivative(z2)
            delta_x1 = w2.T.dot(delta_x2)*ReLU_derivative(z1)

            w4 = w4 - learning_rate * delta_x4.dot(x3.T)
            b4 = b4 - learning_rate * delta_x4
            w3 = w3 - learning_rate * delta_x3.dot(x2.T)
            b3 = b3 - learning_rate * delta_x3
            w2 = w2 - learning_rate * delta_x2.dot(x1.T)
            b2 = b2 - learning_rate * delta_x2
            w1 = w1 - learning_rate * delta_x1.dot(x.T)
            b1 = b1 - learning_rate * delta_x1

            acc = (count_corr / x_train.shape[1])*100
        print(f"Epoch {epoch+1}/{epochs} Accuracy: {acc:.2f}%")
    return w1, b1, w2, b2, w3, b3, w4, b4 

def get_accuracy(x, y, w1, b1, w2, b2, w3, b3, w4, b4):
    _, _, _, _, _, _, _, output = forward_propagetion(w1, b1, w2, b2, w3, b3, w4, b4, x)
    predictions = np.argmax(output, axis=0)
    return np.mean(predictions == y) * 100

model_filename = 'model_weights.npz'
if os.path.exists(model_filename):
    print("Found existing model! Loading weights...")
    saved_model = np.load(model_filename)
    w1 = saved_model['w1']
    b1 = saved_model['b1']
    w2 = saved_model['w2']
    b2 = saved_model['b2']
    w3 = saved_model['w3']
    b3 = saved_model['b3']
    w4 = saved_model['w4']
    b4 = saved_model['b4']
    print("Model loaded successfully.")
else:
    print("No saved model found. Training from scratch...")
    w1, b1, w2, b2, w3, b3, w4, b4 = train(x_train, y_train, epochs=10, learning_rate=0.001)
    np.savez(model_filename, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3, w4=w4, b4=b4)
    print(f"Model saved to {model_filename}.")

dev_accuracy = get_accuracy(
    x_dev, y_dev,
    w1, b1, w2, b2, w3, b3, w4, b4
    )
print(f"Validation Accuracy: {dev_accuracy:.2f}%")

class DrawingApp:
    def __init__(self, root, w1, b1, w2, b2, w3, b3, w4, b4):
        self.root = root
        self.root.title("Draw a digit (0-9)")
        
        self.w1, self.b1 = w1, b1
        self.w2, self.b2 = w2, b2
        self.w3, self.b3 = w3, b3
        self.w4, self.b4 = w4, b4
        
        self.canvas_width = 280
        self.canvas_height = 280
        
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack(pady=10)
        
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        
        self.btn_predict = tk.Button(btn_frame, text="Predict", font=('Arial', 12, 'bold'), command=self.predict_digit)
        self.btn_predict.pack(side=tk.LEFT, padx=10)
        
        self.btn_clear = tk.Button(btn_frame, text="Clear", font=('Arial', 12), command=self.clear_canvas)
        self.btn_clear.pack(side=tk.RIGHT, padx=10)
        
        self.lbl_result = tk.Label(self.root, text="Draw a digit and click Predict", font=('Arial', 14))
        self.lbl_result.pack(pady=10)

    def paint(self, event):
        r = 12
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        

        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.lbl_result.config(text="Canvas cleared.")

    def predict_digit(self):
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        x_input = img_array.reshape(784, 1) / 255.0
        _, _, _, _, _, _, _, x4 = forward_propagetion(self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4, x_input)
        prediction = np.argmax(x4)
        confidence = np.max(x4) * 100
        self.lbl_result.config(text=f"Prediction: {prediction} (Confidence: {confidence:.2f}%)")
        print(f"Model predicted: {prediction} | Confidence: {confidence:.2f}%")

root = tk.Tk()
app = DrawingApp(root, w1, b1, w2, b2, w3, b3, w4, b4)
root.mainloop()
