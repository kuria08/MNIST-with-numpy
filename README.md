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

## Technologies

* Python
* NumPy
* Pandas
* Tkinter
* Pillow
* Matplotlib

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

Training configuration:

```text
Epochs: 10
Learning Rate: 0.01
```

## Run

Install dependencies:

```bash
pip install numpy pandas matplotlib pillow
```

Put the dataset here:

```text
digit-recognizer/train.csv
```

Then run:

```bash
python main.py
```

If no saved model exists, the program trains the neural network automatically. After training, you can draw a digit using the GUI and let the model predict it.

## Goal

This project was created to understand the fundamentals of **Neural Networks and Backpropagation from scratch**, without using TensorFlow or PyTorch.
