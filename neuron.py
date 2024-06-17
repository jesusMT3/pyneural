import activationFunctions as af
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

class Neuron:
    
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
                f"Weights: {self._weights}. "
                f"Bias: {self._bias} ")
        
    def __str__(self):
        return "Neuron()"
        
    # Get and setmethods
    def get_weights(self):
        return self._weights
    
    def set_weights(self, weights: np.array):
        self._weights = weights
    
    def get_bias(self):
        return self._bias
    
    def set_bias(self, bias: float):
        self._bias = bias
    
    def get_activation_function(self):
        return self._activation_function
        
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
    
    n.draw_activation_function(); # Plot activation function