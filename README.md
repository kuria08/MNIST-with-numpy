# MNIST NumPy & Live Canvas

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Power-blue.svg)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green.svg)

A lightweight, 2-layer Neural Network built **entirely from scratch** using only Python and NumPy to recognize handwritten digits. It includes a custom Tkinter GUI for real-time drawing and prediction!

##  Quick Start

1. **Install requirements:** 
   ```bash
   pip install -r requirements.txt
   ```
2. **Setup Data:** Download `train.csv` from Kaggle's Digit Recognizer and place it inside a `digit-recognizer/` folder.
3. **Run:** 
   ```bash
   python main.py
   ```
   *(The model trains once, auto-saves weights to `.npz`, and opens the drawing canvas!)*

##  Key Features

* **Zero Frameworks:** No TensorFlow/PyTorch. Pure linear algebra.
* **Under the Hood:** Explicit Forward/Backpropagation, ReLU, Softmax, and SGD.
* **Live Drawing GUI:** Draw a digit with your mouse and get instant predictions.
* **Auto-Save:** Weights are saved locally after the first run for instant loading in future runs.

##  Architecture & Performance

* **Structure:** `784 (Input)` ➔ `20 (ReLU)` ➔ `10 (Softmax)`
* **Optimizer:** Stochastic Gradient Descent (SGD)
* **Accuracy:** ~96-97% (in just 10 epochs, LR: 0.01)
