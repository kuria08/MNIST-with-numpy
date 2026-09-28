# MNIST Handwritten Digit Recognition — NumPy

A handwritten digit recognition project built **from scratch using NumPy**, with a simple GUI that allows users to draw digits and get predictions from the trained neural network.

## Features

* Neural Network implemented from scratch with NumPy
* ReLU activation
* Softmax output layer
* He initialization
* Forward propagation
* Backpropagation
* Stochastic Gradient Descent (SGD)
* MNIST data preprocessing and normalization
* Train / Validation split
* Save and load trained model using `.npz`
* Interactive Tkinter GUI for drawing digits
* Prediction confidence displayed in real time

## Neural Network

```text
Input: 784
   ↓
512 neurons + ReLU
   ↓
256 neurons + ReLU
   ↓
128 neurons + ReLU
   ↓
10 neurons + Softmax
   ↓
Prediction: 0 - 9
```

The input image is a `28 × 28` grayscale image:

```text
28 × 28 = 784 pixels
```

Pixel values are normalized from:

```text
0 - 255
```

to:

```text
0 - 1
```

## Training

The model is trained using:

* **10 epochs**
* **Learning rate:** `0.001`
* **Optimizer:** Stochastic Gradient Descent
* **Weight initialization:** He Initialization
* **Activation:** ReLU
* **Output activation:** Softmax

The dataset is shuffled using a fixed random seed:

```python
np.random.seed(42)
np.random.shuffle(data)
```

The first `1,000` samples are used for validation, while the remaining samples are used for training.

## Results

Example training result:

```text
Epoch 10/10 Accuracy: 99.82%
Validation Accuracy: 97.40%
```

Training accuracy can vary slightly depending on the trained model, while the fixed random seed keeps the train/validation split consistent.

## GUI

The project includes a simple Tkinter drawing interface.

Users can:

1. Draw a digit from `0` to `9`.
2. Click **Predict**.
3. The neural network predicts the digit.
4. The GUI displays the prediction and confidence.
5. Click **Clear** to draw another digit.

The drawn image is resized from:

```text
280 × 280
```

to:

```text
28 × 28
```

before being passed to the neural network.

## Technologies

* Python
* NumPy
* Pandas
* Tkinter
* Pillow (PIL)
* Matplotlib

## Project Structure

```text
MNIST/
│
├── main.py
├── model_weights.npz
├── digit-recognizer/
│   └── train.csv
│
└── README.md
```

## Installation

Install the required libraries:

```bash
pip install numpy pandas pillow matplotlib
```

Tkinter is included with most standard Python installations on Windows.

## Run

Make sure `train.csv` is located at:

```text
digit-recognizer/train.csv
```

Then run:

```bash
python main.py
```

If `model_weights.npz` already exists, the program loads the saved model instead of training again.

To train a new model, remove:

```text
model_weights.npz
```

and run the program again.

## Goal

This project was built to understand how a neural network works **from the inside**, rather than relying on high-level deep learning frameworks.

The main focus is understanding:

```text
Data
 ↓
Preprocessing
 ↓
Forward Propagation
 ↓
Loss / Gradients
 ↓
Backpropagation
 ↓
Weight Updates
 ↓
Prediction
```

The next step is to further improve the model and experiment with techniques such as mini-batch training, confusion matrix analysis, and CNNs.
