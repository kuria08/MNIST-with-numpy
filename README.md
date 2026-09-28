# Neural Network from Scratch

A 2-layer **Artificial Neural Network (ANN)** built entirely from scratch using **Python** and **NumPy** to recognize handwritten digits (0-9) from the **MNIST dataset**. 

This project operates without deep learning frameworks like TensorFlow or PyTorch, showcasing the mathematical core and raw matrix operations behind neural networks.

## Network Architecture
* **Input Layer:** 784 nodes (28x28 pixel images flattened and normalized).
* **Hidden Layer:** 20 nodes using the **ReLU (Rectified Linear Unit)** activation function.
* **Output Layer:** 10 nodes (representing digits 0-9) using the **Softmax** activation function for multi-class probability distribution.

## Code Architecture & Pipeline
1. **Data Preprocessing:** Loads data via `pandas`, shuffles using `np.random.shuffle`, and normalizes pixel values to a \([0, 1]\) range. Labels are dynamically transformed into \((10 \times 1)\) **One-Hot Vectors** during the training loop.
2. **Forward Propagation:** Matrix dot products (`w.dot(x) + b`) map inputs through the layers, applying sequential ReLU and Softmax activations to compute predictions.
3. **Backpropagation & Training:** Implements explicit **Stochastic Gradient Descent (SGD)**. It computes analytical gradients using the cost derivative and `ReLU_derivative`, manually updating weights and biases at every single sample step.
4. **Interactive Verification:** Integrates a loop utilizing `matplotlib` to render the \(28 \times 28\) grayscale images alongside their real and predicted labels for visual debugging.

## Performance
* Achieves **~96% – 97% Accuracy** in just **10 epochs** using a \(0.01\) learning rate.

## Tech Stack
* Python 3
* NumPy (Matrix operations & linear algebra)
* Pandas (Data pipeline)
* Matplotlib (Visual verification)
