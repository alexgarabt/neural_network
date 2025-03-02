import numpy as np
from typing import Any
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class NeuralNetwork:
    """
    Implementation of the basic vanilla neural network.
    General purpose network that allow the definition of:
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

    def __init__(self, layers: list[int] | None = None, filename:str | None = None) -> None:
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
        # if a file is provided use this parameters
        if filename:
            self.load_model(filename)
        # if layers are provided use this one
        elif layers:
            self.L = len(layers) - 1
            self.weights: dict[int, np.ndarray] =  {}
            self.biases: dict[int, np.ndarray] = {}
            # only used when its needed to save data in a file
            self.layers = layers 

            # Initialize weights and biases using He initialization
            for l in range(1, self.L + 1):
                # He initialization for weights
                self.weights[l] = np.random.randn(layers[l], layers[l-1]) * np.sqrt(2 / layers[l-1])
                self.biases[l] = np.zeros((layers[l], 1))


        else: raise ValueError("Error should provided layers or filename for parameters")

    def save_model(self, filename: str = "parameters.npz", verbose=False):
        """
        Save the current parameters: weights & biases stored in a file

        Parameters
        ----------
        filename : str
            Path and file name, where to save the Parameters
        """
        params = {}
        params["layers"] = self.layers
    
        for l in self.weights:
            params[f"W{l}"] = self.weights[l]
        for l in self.biases:
            params[f"b{l}"] = self.biases[l]
        
        if verbose: print("Saving model Parameters...", end="")
        np.savez(filename, **params)
        if verbose: print(", ...Model saved")
        return

    def load_model(self, filename: str = "parameters.npz", verbose=False):
        """
        Load parameters: weights & biases stored in a file
        """
        loaded_data = np.load(filename)
        self.layers = loaded_data["layers"].tolist()
        self.L = len(self.layers) - 1
        self.weights: dict[int, np.ndarray] =  {}
        self.biases: dict[int, np.ndarray] = {}

        if verbose: print(f"Loading model...", end="")
        for l in range(1, self.L + 1):
            self.weights[l] = loaded_data[f"W{l}"]
            self.biases[l] = loaded_data[f"b{l}"]
            if verbose: print(f"{l/(self.L + 1)*100}%, ", end="")
        if verbose: print()


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
            if l == self.L: A = self.softmax(Z)
            else: A = self.relu(Z)

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
        epsilon = 1e-8 # avoid log(0)
        return (-1/m) * np.sum(Y_G * np.log(Y_P + epsilon))

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

        for l in reversed(range(1, self.L + 1)):
            A_prev = cache[f"A{l-1}"]
            dW = (1/m) * np.dot(dZ, A_prev.T)
            dB = (1/m) * np.sum(dZ, axis=1, keepdims=True)

            grads[f"dW{l}"] = dW
            grads[f"dB{l}"] = dB

            if l > 1:  # Skip for input layer
                dA = np.dot(self.weights[l].T, dZ)

                # Apply the derivative of ReLU
                Z_prev = cache[f"Z{l-1}"]  # Corrected: Use Z{l-1} instead of Z{l}
                dZ = dA * (Z_prev > 0)  # ReLU derivative: 1 if Z > 0, else 0


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

    def batch_generator(self, X: np.ndarray, Y: np.ndarray, batch_size: int):
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
        
        Note
        ----
        If batch_size exceeds the number of samples, the final batch may be smaller 
        than batch_size, containing the remaining samples.
        """
        m = X.shape[1]
        indices = np.arange(m)
        np.random.shuffle(indices)
        for i in range(0, m, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield X[:, batch_indices], Y[:, batch_indices]

    def print_metrics(self, predictions: np.ndarray, true_labels: np.ndarray, cost, epoch=0, batch=0):
        precision = precision_score(true_labels, predictions, average='macro')
        recall = recall_score(true_labels, predictions, average='macro')
        f1 = f1_score(true_labels, predictions, average='macro')
        accuracy = accuracy_score(true_labels, predictions)
   
        print("---------------------------------------------------------------------")
        print(f"Epoch:      {epoch}\n"
              f"Batch:      {batch}\n"
              f"Cost:       {cost:.4f}\n"
              f"Accuracy:   {accuracy:.4f}\n"
              f"Precision:  {precision:.4f}\n"
              f"Recall:     {recall:.4f}\n"
              f"F1 Score:   {f1:.4f}")
        print("---------------------------------------------------------------------")

    def TRAIN(self, X, Y, epochs: int = 1000, learning_rate: float = 0.01, batch_size: int | None = None, verbose: bool = False):
        """
        Train the neural network given a training dataset and the golden labels
        for each input of train set.

        Parameters
        ----------
        X : np.ndarray, shape(input_features, num_samples)
            Training dataset where each column is a sample and each row a feature
        Y: np.ndarray, shape(output_classes, num_samples)
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
            If set, uses mini-batch gradient descent to not iterate over all the training set
            None => Uses all data
        verbose: bool
            If True, prints training metrics
        """
        # Input validation
        if X.shape[0] != self.layers[0]:
            raise ValueError(f"X shape[0] ({X.shape[0]}) must match input layer size ({self.layers[0]})")
        if Y.shape[0] != self.layers[-1]:
            raise ValueError(f"Y shape[0] ({Y.shape[0]}) must match output layer size ({self.layers[-1]})")
        if X.shape[1] != Y.shape[1]:
            raise ValueError(f"Number of samples in X ({X.shape[1]}) and Y ({Y.shape[1]}) must match")

        for i in range(epochs):
            if batch_size:
                batch_gen = self.batch_generator(X, Y, batch_size)
                batch_costs = []
                batch_i = -1
                for X_batch, Y_batch in batch_gen:
                    batch_i += 1
                    Y_P, cache = self.FORWARD_PROPAGATION(X_batch)
                    cost = self.cost_function(Y_P, Y_batch)
                    batch_costs.append(cost)
                    grads = self.BACK_WARDPROPAGATION(X_batch, Y_batch, cache)
                    self.update_parameters(grads, learning_rate)
                    if verbose and batch_i == 0:  # Print only for first batch as an example
                        predictions = np.argmax(Y_P, axis=0)
                        true_labels = np.argmax(Y_batch, axis=0)
                        self.print_metrics(predictions, true_labels, cost, epoch=i, batch=batch_i)
                # Compute and print average cost across batches
                avg_cost = np.mean(batch_costs)
                if verbose:
                    print(f"Epoch {i} - Average Cost Across Batches: {avg_cost:.4f}")
            else:
                Y_P, cache = self.FORWARD_PROPAGATION(X)
                cost = self.cost_function(Y_P, Y)
                grads = self.BACK_WARDPROPAGATION(X, Y, cache)
                self.update_parameters(grads, learning_rate)
                if verbose:
                    predictions = np.argmax(Y_P, axis=0)
                    true_labels = np.argmax(Y, axis=0)
                    self.print_metrics(predictions, true_labels, cost, epoch=i)
        return


    def TEST(self, X, Y, batch_size: int | None = None, verbose=False):
        """
        Allows to test a given set of inputs with its golden labels
        and display
        """
        # Input validation
        if X.shape[0] != self.layers[0]:
            raise ValueError(f"X shape[0] ({X.shape[0]}) must match input layer size ({self.layers[0]})")
        if Y.shape[0] != self.layers[-1]:
            raise ValueError(f"Y shape[0] ({Y.shape[0]}) must match output layer size ({self.layers[-1]})")
        if X.shape[1] != Y.shape[1]:
            raise ValueError(f"Number of samples in X ({X.shape[1]}) and Y ({Y.shape[1]}) must match")

        if batch_size:
            batch_gen = self.batch_generator(X, Y, batch_size)
            batch_i = -1
            for X_batch, Y_batch in batch_gen:
                batch_i += 1
                Y_P, _ = self.FORWARD_PROPAGATION(X_batch)
                cost = self.cost_function(Y_P, Y_batch)
                if verbose:
                    predictions = np.argmax(Y_P, axis=0)
                    true_labels = np.argmax(Y_batch, axis=0)
                    self.print_metrics(predictions, true_labels, cost, batch=batch_i)
        else:
            Y_P, _ = self.FORWARD_PROPAGATION(X)
            cost = self.cost_function(Y_P, Y)
            if verbose:
                predictions = np.argmax(Y_P, axis=0)
                true_labels = np.argmax(Y, axis=0)
                self.print_metrics(predictions, true_labels, cost)
        return
