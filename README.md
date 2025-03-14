# Neural Network
Custom implementation of neural network flexible architecture that allows to define any number of layers and neurons. This repo is a learning proyect.

## Example of use
> [!NOTE]
> In the file `mnist_client.py` is implements the training and testing of the hand-written digit prediction
> In the file `cifar10_client.py` is implements the training and testing of 10 objects image classfication prediction
> The client loads the data, preprocess it and train the NN
The client trains the neural network over the train set and test it in its conterpart. Also implements a function to test the samples over the neural network

### MNIST classification  

https://github.com/user-attachments/assets/dc6c24cb-dd43-4d50-8e02-6fdc8d6c55e1

### CIFAR-10 classification  

https://github.com/user-attachments/assets/4ff68191-21b4-41cb-ba6c-88b6af4b89b9

## Neural Network Implementation

> [!TIP]
> Should look to the `neural_network.py` source content

This repository contains a vanilla neural network implemented from scratch using NumPy. The network is fully customizable, allowing users to define:

- Number of neurons in the **input layer**
- Number of neurons in each **hidden layer**
- Number of neurons in the **output layer**

The model follows standard deep learning techniques, including forward propagation, backpropagation, and gradient descent for parameter optimization.

---

### Architecture & Forward Propagation

Given an input **X**, the network performs forward propagation using the following equations:

For each layer $$l$$:  

$$Z^{(l)} = W^{(l)} A^{(l-1)} + b^{(l)}$$

where:  
- $$Z^{(l)}$$ is the weighted sum of inputs,  

- $$W^{(l)}$$ is the weight matrix,  

- $$A^{(l-1)}$$ is the activation from the previous layer,  

- $$b^{(l)}$$ is the bias term.  

The activation function applied is:  

**ReLU (Rectified Linear Unit) for hidden layers**:  
	$$A^{(l)} = \max(0, Z^{(l)})$$  
**Softmax for output layer**:  
	$$A^{(L)} = \frac{e^{Z^{(L)}}}{\sum e^{Z^{(L)}}}$$  

![Forward propagation](./img/forward.png)
---

### Backpropagation & Parameter Updates

The network uses **cross-entropy loss** for classification:  


```math
\mathcal{L} = -\frac{1}{m} \sum \left( Y \log(A^{(L)}) + (1 - Y) \log(1 - A^{(L)}) \right)
```  

To update the weights and biases, the gradients are computed as follows:  

**Output layer gradient**:  
 $$dZ^{(L)} = A^{(L)} - Y$$  
**Hidden layer gradient**:  
 $$dZ^{(l)} = dA^{(l)} \cdot \mathbb{1}(Z^{(l)} > 0)$$ (ReLU derivative)  
**Gradient of weights and biases**:  
  $$dW^{(l)} = \frac{1}{m} dZ^{(l)} A^{(l-1)T}$$  
  $$db^{(l)} = \frac{1}{m} \sum dZ^{(l)}$$  

The parameters are updated using **gradient descent**:  

$$W^{(l)} = W^{(l)} - \alpha dW^{(l)}$$  
$$b^{(l)} = b^{(l)} - \alpha db^{(l)}$$  

where $$\alpha$$ is the **learning rate**.  

![Backward propagation](./img/backward.png)
---

### API

#### Training & Metrics

- **Training**: The network trains using **mini-batch gradient descent** or **full-batch training** depending on user settings.  
- **Metrics**: During training, key evaluation metrics are computed, including **accuracy, precision, recall, and F1-score**.  

#### How to Use

1. **Initialize the network**:
    ```python
    model = NeuralNetwork(layers=[3, 5, 2])
    ```
2. **Train the model**:
    ```python
    model.TRAIN(X_train, Y_train, epochs=64, learning_rate=0.01, batch_size=32, verbose=True)
    ```
3. **Test the model**:
    ```python
    model.TEST(X_test, Y_test, verbose=True)
    ```
4. **Save & Load parameters**:
    ```python
    model.save_model("parameters.npz")
    model.load_model("parameters.npz")
    ```

This implementation is designed for flexibility, allowing users to train neural networks with varying architectures while providing a simple interface for model training and evaluation.

## Hand-written digit classification implementation
> [!TIP]
> Dataset contents in `dataset/MNIST` path  

For the hand hand-written digit prediction i had used the `MNIST`database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems.

The database contains a total of `70000 samples`

### Training
There are `60000` training samples that where chosen randomly form the original database, the contents are in the file `dataset/MNIST/mnist_train.csv`
```bash
dedalo@lab ~/Projects/vanilla-neural-networks/dataset/MNIST/
$ wc  -l mnist_train.csv
60000 mnist_train.csv
```
---

### Test
There are `10000` test samples that are the remains of the database after substracting the training samples to test real inference of knowledge, the contents are in teh file `dataset/MNIST/mnist_test.csv`
```bash
dedalo@lab ~/Projects/vanilla-neural-networks/dataset/MNIST/
$ wc  -l mnist_test.csv 
10000 mnist_test.csv
```

## CIFAR-10 Classification Implementation

### Dataset

The CIFAR-10 dataset consists of **60000 32×32 color images** categorized into **10 classes**:

- Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

The dataset is structured as follows:
- **Training set**: 50000 images
- **Test set**: 10000 images

Stored in the path: `dataset/CIFAR10/`

### Preprocessing

- Each image is **flattened** into a **3072-dimensional vector** (32×32 pixels × 3 color channels).
- The pixel values are **normalized** to the range `[0,1]` by dividing by 255.
- Labels are **one-hot encoded** into a shape `(10, num_samples)`.


## Dependencies
### Neural Network
The neural network is implemented form scratch, give that the goal was not to rewrite the matrix opeartions or implement the metrics operations the dependencies used are `numpy` for matrix operations and `sklearn` for the testing metrics.

### Training 
The Training dependencies used are `pandas` for reading the dataset and parse it into `numpy` format. `matplotlib` & `pyqpt6` is used for the displaying hand-written digits of the database in graphical way.  
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


