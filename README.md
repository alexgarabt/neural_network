# Neural Network
Custom implementation of the vanilla neural network This repo is a learning proyect.

## Neural Network Implementation

This is a vanilla neural network that uses a fully connected architecture with customizable layers, allowing the definition of:
- \( x \) neurons for the input layer
- \( i_1, i_2, \dots, i_n \) neurons for each hidden layer
- \( y \) neurons for the output layer

The network is designed for general-purpose learning and utilizes the following components:

### Architecture

The neural network is structured as follows:
- **Weights:** \( W^{[l]} \) - weight matrix between layer \( l-1 \) and layer \( l \)
- **Biases:** \( b^{[l]} \) - bias vector for layer \( l \)
- **Activation Values (Cache):**
  - **Pre-activation:** \( Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]} \)
  - **Activation Output:** \( A^{[l]} = f(Z^{[l]}) \)

### Activation Functions

The following activation functions are used:
- **ReLU (Rectified Linear Unit):**
  \[
  f(x) = \max(0, x)
  \]
- **Softmax (for output layer probability normalization):**
  \[
  \sigma(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
  \]

### Forward Propagation

During forward propagation, the network computes activations layer by layer:
\[
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
\]
\[
A^{[l]} = \begin{cases} \text{ReLU}(Z^{[l]}), & \text{if } l < L \\ \text{Softmax}(Z^{[l]}), & \text{if } l = L \end{cases}
\]
where \( L \) is the number of layers in the network.

### Cost Function

The cost function used for optimization is the **Cross-Entropy Loss**:
\[
J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c} + \epsilon)
\]
where:
- \( m \) is the number of training examples
- \( C \) is the number of output classes
- \( y_{i,c} \) is the actual label
- \( \hat{y}_{i,c} \) is the predicted probability for class \( c \)
- \( \epsilon \) is a small constant to prevent log(0)

### Backpropagation

Backpropagation computes gradients to update parameters:
- **Output layer gradient:**
  \[
  dZ^{[L]} = A^{[L]} - Y
  \]
- **Hidden layer gradients:**
  \[
  dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l-1]T}
  \]
  \[
  db^{[l]} = \frac{1}{m} \sum dZ^{[l]}
  \]
  \[
  dZ^{[l-1]} = W^{[l]T} dZ^{[l]} * f'(Z^{[l-1]})
  \]
  where \( f' \) is the derivative of ReLU (1 if \( Z > 0 \), else 0).

### Parameter Update

Using **Gradient Descent**, weights and biases are updated as follows:
\[
W^{[l]} = W^{[l]} - \alpha dW^{[l]}
\]
\[
b^{[l]} = b^{[l]} - \alpha db^{[l]}
\]
where \( \alpha \) is the learning rate.

## API

### Training & Metrics
- **`TRAIN(X, Y, epochs, learning_rate, batch_size, verbose)`**
  - Trains the neural network using forward and backward propagation.
  - Supports mini-batch training.
  - Prints performance metrics if `verbose=True`.

- **`TEST(X, Y, batch_size, verbose)`**
  - Evaluates the trained model using forward propagation.
  - Prints accuracy, precision, recall, and F1-score.

### How to Use

```python
# Initialize a neural network with 3 input neurons, one hidden layer (5 neurons), and 2 output neurons
nn = NeuralNetwork(layers=[3, 5, 2])

# Train the network
nn.TRAIN(X_train, Y_train, epochs=1000, learning_rate=0.01, batch_size=32, verbose=True)

# Test the network
nn.TEST(X_test, Y_test, verbose=True)

# Save the trained model
nn.save_model("model.npz")

# Load a trained model
nn.load_model("model.npz")
```

This neural network provides a flexible and efficient implementation for classification problems, supporting batch training and evaluation with performance metrics.


## MNIST prediction implementation

### Dataset

## Dependencies
### Neural Network
The neural network is implemented form scratch, give that the goal was not to rewrite the matrix opeartions or implement the metrics operations the dependencies used are `numpy` for matrix operations and `sklearn` for the testing metrics.

### Training 
The Training dependencies used for the hand-written digit prediction are `pandas` for reading the dataset and parse it into `numpy` format. `matplotlib` & `pyqpt6` is used for the displaying hand-written digits of the database in graphical way.  
`pyproject.toml`
```toml
[project]
name = "vanilla-neural-networks"
version = "0.1.0"
description = ""
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "matplotlib>=3.10.1",
    "numpy>=2.2.3",
    "pandas>=2.2.3",
    "pyqt6>=6.8.1",
    "scikit-learn>=1.6.1",
]
```


