# MNIST Handwritten Digit Recognition — NumPy

A handwritten digit recognition project built **from scratch using NumPy**, with a simple Tkinter GUI that allows users to draw digits and get predictions from a trained neural network.

The main goal of this project is to understand how a neural network works internally without relying on high-level deep learning frameworks such as TensorFlow or PyTorch.

## Features

* Neural Network implemented from scratch using NumPy
* ReLU activation function
* Softmax output layer
* He weight initialization
* Forward propagation
* Backpropagation
* Stochastic Gradient Descent (SGD)
* MNIST data preprocessing and normalization
* Fixed train / validation split
* Model saving and loading using `.npz`
* Interactive Tkinter GUI
* Direct 28×28 NumPy drawing grid
* Soft brush with grayscale intensity
* Real-time prediction confidence

## Neural Network Architecture

The current neural network has the following architecture:

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

The input image is a:

```text
28 × 28
```

grayscale image.

Therefore:

```text
28 × 28 = 784 pixels
```

Each pixel is represented by a value between:

```text
0 - 255
```

and normalized to:

```text
0 - 1
```

before being passed to the neural network.

## Model Initialization

The network uses **He Initialization** for its weights.

For example:

```python
w1 = np.random.randn(512, 784) * np.sqrt(2 / 784)
```

Biases are initialized to zero:

```python
b1 = np.zeros((512, 1))
```

The same initialization strategy is used for all layers.

## Activation Functions

### ReLU

ReLU is used in the three hidden layers:

```python
def ReLU(x):
    return np.maximum(0, x)
```

The derivative used during backpropagation is:

```python
def ReLU_derivative(x):
    return x > 0
```

### Softmax

The output layer uses Softmax to convert the final outputs into probabilities:

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)
```

The class with the highest probability is selected as the prediction.

## Training

The model is trained using:

* **Epochs:** 10
* **Learning rate:** `0.001`
* **Optimizer:** Stochastic Gradient Descent
* **Weight initialization:** He Initialization
* **Hidden activation:** ReLU
* **Output activation:** Softmax

Each training sample is processed individually:

```text
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Error
  ↓
Backpropagation
  ↓
Weight Update
```

The weights and biases are updated after each individual training sample.

## Dataset

This project uses the **MNIST handwritten digit dataset** from the Kaggle Digit Recognizer dataset.

The training data is stored in:

```text
digit-recognizer/train.csv
```

The dataset is shuffled using a fixed random seed:

```python
np.random.seed(42)
np.random.shuffle(data)
```

The data is then divided into:

```text
1,000 samples
    ↓
Validation set

Remaining samples
    ↓
Training set
```

This makes the train/validation split reproducible.

## One-Hot Encoding

Labels from `0` to `9` are converted into one-hot vectors.

For example:

```text
Label: 3
```

becomes:

```text
[0]
[0]
[0]
[1]
[0]
[0]
[0]
[0]
[0]
[0]
```

The resulting vector has shape:

```text
10 × 1
```

## Results

Example training result:

```text
Epoch 10/10 Accuracy: 99.82%
Validation Accuracy: 97.40%
```

The exact training and validation accuracy may vary depending on the trained model.

The validation split itself remains consistent because the dataset is shuffled using:

```python
np.random.seed(42)
```

## Model Saving and Loading

After training, the model parameters are saved to:

```text
model_weights.npz
```

The file contains:

```text
w1
b1
w2
b2
w3
b3
w4
b4
```

When the program starts, it checks whether the model file already exists.

If it exists:

```text
Load existing model
        ↓
Skip training
        ↓
Evaluate validation accuracy
        ↓
Start GUI
```

If it does not exist:

```text
Train model
     ↓
Save weights
     ↓
Evaluate validation accuracy
     ↓
Start GUI
```

To train a new model, delete:

```text
model_weights.npz
```

and run the program again.

## GUI

The project includes an interactive Tkinter drawing interface.

Users can:

1. Draw a handwritten digit.
2. Click **Predict**.
3. The neural network predicts the digit.
4. The GUI displays the predicted digit and confidence.
5. Click **Clear** to erase the drawing.

### Direct 28×28 Drawing

Instead of drawing on a 280×280 image and resizing it afterward, the GUI internally stores the drawing directly as a:

```text
28 × 28
```

NumPy array.

The canvas is only enlarged by a factor of 10 for easier drawing:

```text
28 × 28 internal grid
        ↓
280 × 280 displayed canvas
```

Therefore, the model receives the original 784-pixel representation directly.

### Soft Brush

The GUI uses a soft brush instead of simply turning pixels completely on or off.

The intensity depends on the distance from the center of the brush:

```text
Center → stronger intensity
Edge   → weaker intensity
```

The intensity is accumulated in the internal NumPy array and clipped to a maximum value of `1.0`.

This produces a smoother grayscale representation of the handwritten digit.

## Prediction

When the user clicks **Predict**, the 28×28 grid is converted into a 784×1 input vector:

```python
x_input = self.grid.reshape(784, 1)
```

The input is then passed through the neural network.

The predicted digit is obtained using:

```python
prediction = np.argmax(x4)
```

The confidence displayed by the GUI is the highest Softmax output:

```python
confidence = np.max(x4) * 100
```

Example:

```text
Prediction: 7
Confidence: 98.42%
```

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

Install the required Python packages:

```bash
pip install numpy pandas pillow matplotlib
```

Tkinter is included with most standard Python installations on Windows.

## Run

Make sure the dataset is located at:

```text
digit-recognizer/train.csv
```

Then run:

```bash
python main.py
```

If `model_weights.npz` already exists, the program loads the saved model instead of training again.

To train a new model:

1. Delete `model_weights.npz`.
2. Run the program again.

## Learning Objective

This project was built to understand the internal mechanics of a neural network rather than simply using a pre-built deep learning framework.

The main learning process is:

```text
MNIST Dataset
      ↓
Data Preprocessing
      ↓
Normalization
      ↓
One-Hot Encoding
      ↓
Forward Propagation
      ↓
Softmax
      ↓
Prediction
      ↓
Error
      ↓
Backpropagation
      ↓
Gradient Calculation
      ↓
Weight & Bias Updates
      ↓
Trained Neural Network
      ↓
Digit Prediction
```

## Future Improvements

Possible improvements for future versions include:

* Mini-batch Gradient Descent
* Better training efficiency
* Learning rate scheduling
* Confusion matrix analysis
* Per-class accuracy analysis
* Data augmentation
* CNN implementation from scratch
* Comparison between fully connected networks and CNNs
* Improved GUI drawing and preprocessing
* Testing on external handwritten digit images

## Disclaimer

This project is primarily an educational implementation designed to understand the fundamentals of neural networks, forward propagation, backpropagation, and gradient-based optimization using NumPy.
