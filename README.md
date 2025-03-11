# Neural Network
Custom implementation of the vanilla neural networks, gradient descent & backpropagation.
This repo is a learning proyect.

## Neural Network Implementation

This is a vanilla feedforward neural network implemented from scratch in Python using NumPy. It supports multi-layer architectures with ReLU activation for hidden layers and softmax for the output layer, optimized using gradient descent with cross-entropy loss. Below is a detailed breakdown of its implementation, including the mathematical foundations.

### Architecture
The network is fully connected with a configurable number of layers and neurons:
- **Input Layer**: Matches the number of input features ($n^{[0]}$), e.g., 784 for MNIST (28×28 pixels).
- **Hidden Layers**: User-defined number of layers ($L-1$) and neurons per layer ($n^{[l]}$), using ReLU activation.
- **Output Layer**: Matches the number of classes ($n^{[L]}$), e.g., 10 for MNIST, with softmax activation.

Parameters:
- **Weights** ($W^{[l]}$): For layer $l$, shape $(n^{[l]}, n^{[l-1]})$, initialized with He initialization: $W^{[l]} \sim \mathcal{N}(0, \sqrt{2 / n^{[l-1]}})$.
- **Biases** ($b^{[l]}$): Shape $(n^{[l]}, 1)$, initialized to zeros.

### Forward Propagation
Forward propagation computes the activations layer by layer for an input $X$ of shape $(n^{[0]}, m)$, where $m$ is the number of samples.

For each layer $l$:
- **Pre-activation**: 
  $$
  Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}
  $$
  where $A^{[0]} = X$.
- **Activation**:
  - Hidden layers ($l = 1, ..., L-1$):
    $$
    A^{[l]} = \text{ReLU}(Z^{[l]}) = \max(0, Z^{[l]})
    $$
  - Output layer ($l = L$):
    $$
    A^{[L]} = \text{softmax}(Z^{[L]}) = \frac{e^{Z^{[L]} - \max(Z^{[L]})}}{\sum_{j} e^{Z^{[L)}_j - \max(Z^{[L]})}}
    $$
    (Subtracting $\max(Z^{[L]})$ ensures numerical stability.)

The final output $A^{[L]}$ represents class probabilities, shape $(n^{[L]}, m)$. Intermediate $Z^{[l]}$ and $A^{[l]}$ values are cached for backpropagation.

### Cost Function
The network uses cross-entropy loss to measure the difference between predicted probabilities ($A^{[L]}$) and true labels ($Y$), both of shape $(n^{[L]}, m)$:
$$
J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{n^{[L]}} Y_{k,i} \log(A^{[L]}_{k,i} + \epsilon)
$$
- $Y_{k,i}$: True label (1 if class $k$ is correct, 0 otherwise).
- $A^{[L]}_{k,i}$: Predicted probability for class $k$, sample $i$.
- $\epsilon = 10^{-8}$: Small constant to avoid $\log(0)$.

### Backpropagation
Backpropagation computes gradients of the cost $J$ with respect to parameters, propagating errors backward:
- **Output Layer ($l = L$)**:
  - For softmax with cross-entropy, the gradient simplifies to:
    $$
    dZ^{[L]} = A^{[L]} - Y
    $$
- **Hidden Layers ($l = L-1, ..., 1$)**:
  - Gradient w.r.t. activation:
    $$
    dA^{[l]} = W^{[l+1]T} \cdot dZ^{[l+1]}
    $$
  - Gradient w.r.t. pre-activation (ReLU derivative):
    $$
    dZ^{[l]} = dA^{[l]} \cdot \mathbb{1}(Z^{[l]} > 0)
    $$
    where $\mathbb{1}(Z^{[l]} > 0)$ is 1 if $Z^{[l]} > 0$, 0 otherwise.

- **Gradients for Parameters**:
  - Weights:
    $$
    dW^{[l]} = \frac{1}{m} dZ^{[l]} \cdot A^{[l-1]T}
    $$
  - Biases:
    $$
    db^{[l]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[l]}_{:,i}
    $$

### Training Process
The network is trained using gradient descent, with optional mini-batch support:
- **Full Batch**: Updates parameters using the entire dataset ($X$, $Y$).
- **Mini-Batch**: Shuffles data and processes batches of size `batch_size`, yielding $X_{\text{batch}}$, $Y_{\text{batch}}$.

For each epoch:
1. Forward propagation computes $A^{[L]}$ and caches intermediates.
2. Backpropagation calculates gradients $dW^{[l]}$, $db^{[l]}$.
3. Parameters are updated:
   $$
   W^{[l]} \gets W^{[l]} - \alpha \cdot dW^{[l]}
   $$
   $$
   b^{[l]} \gets b^{[l]} - \alpha \cdot db^{[l]}
   $$
   where $\alpha$ is the learning rate (default 0.01).

- **Metrics**: When `verbose=True`, accuracy, precision, recall, and F1-score are computed using scikit-learn, with predictions derived via $\arg\max(A^{[L]})$.

### Implementation Details
- **Initialization**: He initialization for weights reduces vanishing gradient issues with ReLU.
- **Mini-Batch**: Random shuffling ensures varied gradient updates.
- **Model Persistence**: `save_model` and `load_model` store/load parameters in `.npz` files.
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
