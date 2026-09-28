# MNIST Classification with NumPy (From Scratch) 

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Power-blue.svg)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green.svg)

A neural network built **entirely from scratch** using only Python and NumPy to recognize handwritten digits. No deep learning frameworks were used, highlighting the raw linear algebra and calculus behind deep learning.

##  Architecture & Performance
* **Structure:** `784` (Input) $\rightarrow$ `20` nodes + ReLU (Hidden) $\rightarrow$ `10` nodes + Softmax (Output).
* **Optimizer:** Stochastic Gradient Descent (SGD) with a Learning Rate of `0.01`.
* **Accuracy:** Consistently achieves **~96-97% accuracy** on the training set within 10 epochs.

##  Interactive Live Canvas
The `DrawingApp` class uses **Tkinter** and **Pillow (PIL)** to create a drawing board. 
When you draw a number and hit *Predict*, the app:
1. Resizes the drawing to $28 \times 28$.
2. Flattens it into a $784 \times 1$ matrix.
3. Feeds it directly into the custom `forward_propagetion()` function to output real-time predictions.

##  Getting Started

1. **Install dependencies:** `pip install numpy pandas matplotlib Pillow`
2. **Data:** Download `train.csv` from Kaggle's Digit Recognizer and put it in the `digit-recognizer/` folder.
3. **Run:** Execute `python main.py`. (The model trains on the first run, saves weights to `model_weights.npz`, and then pops up the drawing GUI!).
