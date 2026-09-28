import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = np.array(pd.read_csv('digit-recognizer/train.csv'))

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) #tron data

#data for dev
data_dev = data[0:1000].T
y_dev = data_dev[0]
x_dev = data_dev[1:n] / 255.

#data for train
data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n] / 255.

#chuyen labels 0->9 thanh one hot vector (10x1)
def one_hot(label):
    one_hot = np.zeros((10, 1))
    one_hot[label] = 1
    return one_hot

def init():
    w1 = np.random.randn(20, 784)*0.01
    b1 = np.zeros((20, 1))
    w2 = np.random.randn(10, 20)*0.01
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

#activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#cost function
def cost(out_put, labels):
    return (1/len(out_put))*np.sum((out_put-labels)**2)


def forward_propagetion(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    x1 = sigmoid(z1) 
    z2 = w2.dot(x1) + b2
    x2 = sigmoid(z2) #output
    return x1, x2

#training
def train(x_train, y_train, epochs, learning_rate):
    w1, b1, w2, b2 = init()
    for epoch in range(epochs):
        count_corr = 0
        for i in range(x_train.shape[1]):
            labels = one_hot(y_train[i])
            x = x_train[:,i:i+1]
            x1, x2 = forward_propagetion(w1, b1, w2, b2, x)

            count_corr += int(np.argmax(x2) == np.argmax(labels))

            delta_x2 = (x2 - labels)
            delta_x1 = (w2.T.dot(delta_x2))*(x1*(1-x1))

            w2 = w2 - learning_rate*delta_x2.dot(x1.T)
            b2 = b2 - learning_rate*delta_x2
            w1 = w1 - learning_rate*delta_x1.dot(x.T)
            b1 = b1 - learning_rate*delta_x1
            acc = (count_corr / x_train.shape[1])*100
        print(f"Epoch {epoch+1}/{epochs} Accuracy: {acc:.2f}%")
    return w1, w2, b1, b2
w1, w2, b1, b2 = train(x_train, y_train, epochs=10, learning_rate=0.01)

def test(w1, b1, w2, b2, x_dev, y_dev):
    while True:
        inp = input(f"\nType 0 -> {x_dev.shape[1] - 1}, press 'q' for exit): ")
        
        if inp.lower() == 'q':
            break
        idx = int(inp)
        if not inp.isdigit() or idx < 0 or idx > x_dev.shape[1] - 1:
            print("Try again!")
            continue

        x = x_dev[:, idx:idx+1]
        act = y_dev[idx]

        _, x2 = forward_propagetion(w1, b1, w2, b2, x)
        pre = np.argmax(x2)

        print(f"Predict: {pre}")
        print(f"Real   : {act}")

        img_display = x.reshape(28, 28) * 255.0
        plt.figure(figsize=(4, 4))
        plt.imshow(img_display, cmap='gray')
        plt.title(f"Index: {idx} | Prediction: {pre} | Actual: {act}")
        plt.axis('off')
        plt.show()

test(w1, b1, w2, b2, x_dev, y_dev)
