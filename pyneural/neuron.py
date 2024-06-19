"""
Empty docstring (for now)
"""

import activation_functions as af
import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    """
    The neuron class will create objects that follow the conventions for the 
    common neuron architecture. They will have weights for each of their inputs
    as a numpy array, as well as a bias. 
    
    The activation function will be passed in string form, and the class will 
    look into the activation functions moduleto find one that meets the string. 
    Some of the examples are shown below:
        - Sigmoid function: "sigmoid".
        - Rectified linear unit function: "relu".
        - Gaussian function: "gaussian".
  
    To create a neuron, it is important to know how many inputs will it have
    in order to create the array of weights. The following code is to initalize a 
    neuron with 4 inputs, each weight of value 1 and bias of value 0, with
    the activation function as the rectified linear unit function:
    
    >>> neuron = Neuron(weights = [1, 1, 1, 1], 
    >>>                            bias = 0, 
    >>>                            activation_function = "relu")
    
    The attributes of a neuron are intended to be private, and "set" and "get" functions
    are developed to modify the following attributes:
        - Weights: get_weights(), set_weights(list)
        - Bias: get_bias(), set_bias(float)
        - Activation function: get_activation_function(), set_activation_function(str)
    """
    def __init__(self, weights: np.array, bias: float, activation_function: str = "sigmoid"):
        """
        Neuron class

        Args:
            weights (np.array): Array of weights comming from the neuron
            bias (float): Bias of the neuron
            activationFunction (str): Specify the activation function
        """
        self._weights = weights
        self._bias = bias
        self._activation_function = activation_function

    def __repr__(self):
        return (f"\n Neuron(). "
                f"Weights: {self.get_weights()}. "
                f"Bias: {self.get_bias()} "
                f"Activation function: {self.get_activation_function()}")

    def __str__(self):
        return "Neuron()"

    # Get and setmethods
    def get_weights(self):
        return self._weights

    def set_weights(self, weights: list):
        self._weights = weights

    def get_bias(self):
        return self._bias

    def set_bias(self, bias: float):
        self._bias = bias

    def get_activation_function(self):
        return self._activation_function
    
    def set_activation_function(self, activation_function: str):
        self._activation_function = activation_function

    def feedforward(self, inputs):
        total = np.dot(self._weights, inputs) + self._bias
        activation_function = getattr(af, self._activation_function)
        return activation_function(total)

    # Other methods
    def draw_activation_function(self):
        """Plots the activation function"""

        x = [round(i * 0.1, 1) for i in range(-100, 101)]
        activation_function = getattr(af, self._activation_function)
        y = [activation_function(i) for i in x]
        plt.title(f"Activation function: {self.get_activation_function()}")
        plt.grid(True)
        plt.plot(x, y)
        plt.show()        

if __name__ == "__main__":
    weights = np.array([0, 1]) # w1 = 0, w2 = 1
    bias = 4                   # b = 4
    n = Neuron(weights, bias, activation_function = "sigmoid")

    x = np.array([2, 3])       # x1 = 2, x2 = 3
    print(n.feedforward(x))

    n.draw_activation_function() # Plot activation function
