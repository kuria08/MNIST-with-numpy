# MNIST Digit Recognition with NumPy

A simple handwritten digit recognition project built from scratch using **Python and NumPy**.

## Features

* Neural Network: `784 → 20 → 10`
* ReLU activation
* Softmax output
* Backpropagation
* Gradient Descent
* MNIST data preprocessing
* Model saving/loading
* GUI for drawing digits and making predictions

## Model

```text
784 Input
   ↓
20 Hidden Neurons
   ↓
ReLU
   ↓
10 Output Neurons
   ↓
Softmax
```

## Results

* **Accuracy: ~96–97%**
* **Epochs: 10**
* **Learning rate: 0.01**

## Technologies

* Python
* NumPy
* Pandas
* Tkinter
* Pillow
* Matplotlib

## Run

```bash
pip install numpy pandas matplotlib pillow
python main.py
```

Make sure `train.csv` is located at:

```text
digit-recognizer/train.csv
```

The program automatically trains the model if no saved weights are found. After training, you can draw a digit using the GUI and let the model predict it.

## Goal

Built to understand **Neural Networks, Forward Propagation, Backpropagation, and Gradient Descent from scratch**, without TensorFlow or PyTorch.
