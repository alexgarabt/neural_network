import numpy as np

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
