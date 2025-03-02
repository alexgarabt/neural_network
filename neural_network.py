import numpy as np
from typing import Any
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class NeuralNetwork:
    """
    Implementation of the basic vanilla neural network.
    General porpuse network that allow the definition of:
        - (x) neurons for the input layer, 
        - (i1,i2,...,in) neurons each for each hidden layer,
        - (y) neurons for the output layer,

    Parameters:
        - weights => list[l, matrix[NxN weights Layer(l-1)<=>Layer(l)]]
        - bias => list[l, matrix[1xN of l(layer)]]
        - activations/z => (CACHE) last one:
                - Z => dict["Zi" (for) L, Wi*Ai-1 + bi]
                - A => dict["Ai" (for) L, sigmoid(Ai)]

    Non-linear functions:
        - ReLU(x) => max(0, x)

    Output function (transform activation values to probabilities):
        - softmax(X) =>  [(xi)/(sum(xj are in X))]
    """

    def __init__(self, layers: list[int]) -> None:
        """
        Initialize the neural network with as many layers as (layers) length
        and with as many neurons as defined for each one in its position.
        Also initialize the weight to random values and biases to 0

        Parameters
        ----------
        layers: list[int]
            layers[0]   => how many neurons are in the INPUT LAYER
            layers[i]   => how many neurons are in the ITH HIDDEN LAYER
            layers[-1]  => how many neurons are in the OUTPUT LAYER

            Example: [3, 5, 2] (Input layer: 3 neurons, Hidden: 5, Output: 2)
        """
        self.L = len(layers) - 1
        self.weights: dict[int, np.ndarray] =  {}
        self.biases: dict[int, np.ndarray] = {}

        # Initialize weights and biases
        for l in range(1, self.L + 1):
            # weight random values to break simetry
            # small variations to avoid large activation that leads to saturation of activation function
            self.weights[l] = np.random.rand(layers[l], layers[l-1]) * 0.01
            self.biases[l] = np.zeros((layers[l], 1))

    def relu(self, Z: np.ndarray) -> np.ndarray:
        """Implementation of ReLU non-linear function for a given matrix"""
        return np.maximum(0, Z)

    def softmax(self, Z: np.ndarray) -> np.ndarray:
        """Implementation of the softmax function for normalize values"""
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return (expZ / np.sum(expZ, axis=0, keepdims=True))

    def FORWARD_PROPAGATION(self, X: np.ndarray) -> tuple[np.ndarray,dict[str, np.ndarray]]:
        """
        Inference or forward propagation of the neuron actual paramentes given an input values

        Parameters
        ----------
        X : np.ndarray
            Values for the input layer of the neural network
            Shape should be accord with the input layer shape
        
        Returns
        -------
        A : Output layer activation values => PROBABILITIES
        cache : The cached values of (activations/z)
           {"Ai": array with activation values for layer (i)} 
           {"Zi": array with z values for layer (i)} 
        """
        A = X
        cache = {"A0": A}

        for l in range(1, self.L + 1):
            # z = w(i)*a(i-1) + b(i)
            Z = np.dot(self.weights[l], A) + self.biases[l]

            # a = non-linear(z)
            if l == self.L: A = self.relu(Z)
            else: A = self.softmax(Z)

            # cache the activation results
            cache[f"A{l}"] = A
            cache[f"Z{l}"] = Z

        return A, cache

    def cost_function(self, Y_P:np.ndarray, Y_G:np.ndarray):
        """
        Compute cost using CROSS-ENTROPY LOSS, not minimum square difference

        Parameters
        ----------
            Y_P: Predicted output
            Y_G: Golden label output
        """
        m = Y_G.shape[1]
        return (-1/m) * np.sum(Y_G * np.log(Y_P))

    def BACK_WARDPROPAGATION(self, X: np.ndarray, Y: np.ndarray, cache:dict[str, np.ndarray]) -> dict[str, Any] :
        """
        Perform the back_propagation of the neural network given the input values,
        the golden label output values and the cache for (activations/z)

        Parameters
        ----------
        X : np.ndarray
            Input layer values
        Y: np.ndarray
            Golden label output values
        cache : The cached values of (activations/z)
           {"Ai": array with activation values for layer (i)} 
           {"Zi": array with z values for layer (i)}  

        Return
        ------
        The gradients calculated
            {"dWi": difference of weights in layer (i)}
            {"dBi": difference of biases in layer (i)}
        """
        m = X.shape[1]
        grads = {}

        # Compute the output layer gradient
        A_final = cache[f"A{self.L}"]
        dZ = A_final - Y # softmax derivate for cross-entropy loss

        for l in reversed(range(1, self.L +1)):
            A_prev = cache[f"A{l-1}"]
            dW = (1/m) * np.dot(dZ, A_prev.T)
            dB = (1/m) * np.sum(dZ, axis=1, keepdims=True)

            grads[f"dW{l}"] = dW
            grads[f"dB{l}"] = dB

            # back propagation with ReLU
            if l > 1:
                dA = np.dot(self.weights[l].T, dZ)
                dZ = np.where(cache[f"Z{l-1}"] > 0, dA, 0)

        return grads

    def update_parameters(self, grads: dict[str, Any], learning_rate=0.01):
        """
        Update the weight/biases parameters of the neural network 
        using the gradient descent values.

        Parameters
        ----------
        grads : dict[str, Any]
            Calculated gradient values
            {"dWi": difference of weights in layer (i)}
            {"dBi": difference of biases in layer (i)}
        """
        for l in range(1, self.L +1):
            self.weights[l] -= learning_rate * grads[f"dW{l}"]
            self.biases[l] -= learning_rate * grads[f"dB{l}"]

    def batch_generator(self, X: np.ndarray, Y:np.ndarray, batch_size: int):
        """
        Generator function to yield mini-batches of data.
        
        Parameters
        ----------
        X : np.ndarray 
            Input data of shape (features, samples)
        Y : np.ndarray 
            True labels of shape (classes, samples)
        batch_size : int
             Size of each mini-batch
        
        Yields
        ------
        Tuple[np.ndarray, np.ndarray]: Mini-batch (X_batch, Y_batch)
        """ 
        m = X.shape[1]
        indices = np.arange(m)
        np.random.shuffle(indices)
        for i in range(0, m, batch_size):
            batch_indices = indices[i: i+batch_size]
            yield X[: batch_indices], Y[:, batch_indices]

    def print_metrics(self, predictions: np.ndarray, true_labels: np.ndarray, cost, epoch=0, batch=0):
        precision = precision_score(true_labels, predictions, average='macro')
        recall = recall_score(true_labels, predictions, average='macro')
        f1 = f1_score(true_labels, predictions, average='macro')
        accuracy = accuracy_score(true_labels, predictions)
   
        print("---------------------------------------------------------------------")
        print(f"Epoch:      {epoch}, 
                Batch:      {batch}
                Cost:       {cost:.4f}, 
                Accuracy:   {accuracy:.4f}, 
                Precision:  {precision:.4f}, 
                Recall:     {recall:.4f}, 
                F1 Score:   {f1:.4f}")
        print("---------------------------------------------------------------------")


    def TRAIN(self, X, Y, epochs: int=1000, learning_rate:float =0.01, batch_size:int | None = None, verbose: bool = True):
        """
        Train the neural network given a training dataset and the golden labels
        for each input of train set.

        Parameters
        ----------
        X : np.ndarray, shape(input_features, num_samples)
            Training dataset where each column is a sample and each row a feature
        Y: np.ndarray, shape(output_clases, num_samples)
            Golden label/True label corresponding to each sample of the training set
        epochs: int
            Number of times the entire dataset is passed through the neural network
            May lead to overfit
        learning_rate: float
            Control of the step size while updating the network parameters
            (Too high => May overshoot optimal values)
            (Too low => Slow convergence)
        batch_size: int
            Controls the number of randomly selected training samples per update.
            if set uses mini-batch gradient descent to not iterate over all the training set
            None => Uses all data
        """
        cost = 0
        for i in range(epochs):
            if batch_size:
                batch_gen = self.batch_generator(X,Y,batch_size)
                batch_i = -1
                for X_batch, Y_batch in batch_gen:
                    batch_i += 1
                    Y_P, cache = self.FORWARD_PROPAGATION(X_batch)
                    cost = self.cost_function(Y_P, Y_batch)
                    grads = self.BACK_WARDPROPAGATION(X_batch, Y_batch, cache)
                    self.update_parameters(grads, learning_rate)
                    if verbose:
                        predictions = np.argmax(Y_P, axis=0)
                        true_labels = np.argmax(Y, axis=0)
                        self.print_metrics(predictions, true_labels, cost, epoch=i, batch=batch_i)
            else:
                Y_P, cache = self.FORWARD_PROPAGATION(X)
                cost = self.cost_function(Y_P, Y)
                grads = self.BACK_WARDPROPAGATION(X, Y, cache)
                self.update_parameters(grads, learning_rate)
                if verbose:
                    predictions = np.argmax(Y_P, axis=0)
                    true_labels = np.argmax(Y, axis=0)
                    self.print_metrics(predictions, true_labels, cost, epoch=i)


