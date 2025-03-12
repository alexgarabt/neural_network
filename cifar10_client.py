import numpy as np
from matplotlib import pyplot as plt
from neural_network import NeuralNetwork

"""
Implementing the classification of CIFAR-10 dataset
Reference: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. 
The test batch contains exactly 1000 randomly-selected images from each class. 
The training batches contain the remaining images in random order, but some training batches may 
contain more images from one class than another. Between them, the training batches contain 
exactly 5000 images from each class.

dataset path => ./dataset/CIFAR10/

Data shape => (10000, 3072) 10000 images of 32x32 pixels each of 3 colors (RGB)
Labels shape => (10000,)    10000 labels each one for an image
"""

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load CIFAR-10 data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic

def load_cifar10_batch(file):
    batch = unpickle(file)
    X = batch[b'data'].astype(np.float32)
    Y = np.array(batch[b'labels'])
    return X, Y

# Load training data
train_files = [f"dataset/CIFAR10/data_batch_{i}" for i in range(1, 6)]
X_train_batches = []
Y_train_batches = []

for file in train_files:
    X_batch, Y_batch = load_cifar10_batch(file)
    X_train_batches.append(X_batch)
    Y_train_batches.append(Y_batch)

X_train = np.concatenate(X_train_batches, axis=0)  # (50000, 3072)
Y_train = np.concatenate(Y_train_batches, axis=0)  # (50000,)

# Load test data
X_test, Y_test = load_cifar10_batch("dataset/CIFAR10/test_batch")  # (10000, 3072), (10000,)

# Preprocess
X_train = X_train.T / 255.0  # (3072, 50000)
Y_train = np.eye(10)[Y_train].T  # (10, 50000)
X_test = X_test.T / 255.0    # (3072, 10000)
Y_test = np.eye(10)[Y_test].T  # (10, 10000)

# Initialize the network
nn = NeuralNetwork(layers=[3072, 256, 128, 10])
# 32pixels x 32pixels x 3(RGB) => 3072 input neurons
# 2 hidden layers => 256 neurons, 128 neurons (gradual reduction)
# output classes => 10 neurons each for type of class

# Train
nn.TRAIN(X_train, Y_train, epochs=20, batch_size=64, verbose=True)

print("-----------------------------------------------")
print("TEST METRICS")
nn.TEST(X_test, Y_test, verbose=True)
print("-----------------------------------------------")

# Prediction function with image display
# Poor quality of images due to be 32x32 pixels only
def test_prediction_testsamples(index):
    current_image = X_test[:, index, None]  # Shape: (3072, 1)
    Y_P, _ = nn.FORWARD_PROPAGATION(current_image)
    predicted_class_index = np.argmax(Y_P, axis=0)[0]
    predicted_class_name = CIFAR10_CLASSES[predicted_class_index]
    print(f"Predicted class: {predicted_class_name} (index: {predicted_class_index})")
    print("Prediction probabilities:\n", Y_P)
    # Extract flat image and scale back
    flat_image = current_image[:, 0] * 255  # Shape: (3072,)
    # Reshape to (3, 32, 32) directly (channels first)
    image_3d = flat_image.reshape(3, 32, 32)  # Shape: (3, 32, 32)
    # Transpose to (32, 32, 3) for matplotlib
    current_image = np.transpose(image_3d, (1, 2, 0))  # Shape: (32, 32, 3)
    # show the image and improve show with interpolation
    plt.imshow(current_image.astype(np.uint8), interpolation="lanczos")
    plt.show()
