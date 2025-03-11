import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from neural_network import NeuralNetwork

# Load the data
train_data = pd.read_csv("./dataset/mnist_train.csv")
test_data = pd.read_csv("./dataset/mnist_test.csv")

train_data = np.array(train_data)
test_data = np.array(test_data)
m_train, n_train = train_data.shape
m_test, n_test = test_data.shape

# TRAIN DATA
train_data = train_data.T
Y_train = train_data[0]  # Shape: (m_train,)
X_train = train_data[1:]  # Shape: (784, m_train)
X_train = X_train / 255.  # Normalize to [0, 1]
Y_train = np.eye(10)[Y_train].T  # One-hot: (10, m_train)

# TEST DATA
test_data = test_data.T
Y_test = test_data[0]  # Shape: (m_test,)
X_test = test_data[1:]  # Shape: (784, m_test)
X_test = X_test / 255.  # Normalize to [0, 1]
Y_test = np.eye(10)[Y_test].T  # One-hot: (10, m_test)

# Initialize the network
nn = NeuralNetwork(layers=[28*28, 128, 128, 10])
# 784 input neurons, one for each pixel
# 2 hidden layers of 128 neurons
# 10 output neurons, one for each class/number

# Train with more epochs and a reasonable batch size
nn.TRAIN(X_train, Y_train, epochs=20, batch_size=64, verbose=True)

print("-----------------------------------------------")
print("TEST METRICS")
# Test the network
nn.TEST(X_test, Y_test, verbose=True)
print("-----------------------------------------------")

def test_prediction_testsamples(index):
    current_image = X_test[:, index, None]
    Y_P, _ = nn.FORWARD_PROPAGATION(X_test[:, index, None])
    print(f"God sait its => {np.argmax(Y_P, axis=0)}")
    print("Predictions: \n", Y_P)
    current_image = current_image.reshape((28,28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation="nearest")
    plt.show()


