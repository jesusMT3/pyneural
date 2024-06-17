from neuron import Neuron
import numpy as np
from dataclasses import dataclass

class MyNeuralNetwork:
    """
    A neural network with:
        - 2 inputs
        - a hidden layer with 2 neurons (h1, h2)
        - an output layer with 1 neuron (o1)
  Each neuron has the same weights and bias:
        - w = [0, 1]
        - b = 0
    """
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0
        
        # Hidden layer and output layers are neuron objects
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)
        
    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        
        return out_o1
    
# Can I generalize this?
class NeuralNetwork:
    def __init__(self, config: np.array,  activation_function: str = "sigmoid"):
        self._n_hidden_layers = len(config[:-1])
        self._hidden_layers = config[1:-1]
        self._input_layer = config[0]
        self._output_layer = config[-1]
        self._neurons = []
        
        for i, num_neurons in enumerate(config):
            if i == 0:
                continue # TODO: Fix this
            else:
                print(f"Layer #{i}: Hidden layer")
                weights = np.ones(config[i-1])
                bias = 0
                layer = [Neuron(weights, bias) for _ in range(num_neurons)] 
                self._neurons.append(layer) 
        
    # Get methods
    
    def get_hidden_layers(self):
        return self._hidden_layers
        
    def get_n_hidden_layers(self):
        return self._n_hidden_layers
    
    def get_input_layer(self):
        return self._input_layer
    
    def get_output_layer(self):
        return self._output_layer
    
    # TODO: Change this to repr o str method
    def print_configuration(self): 
        print(f"""
Input layer: {self.get_input_layer()}. 
Hidden layers: {self.get_hidden_layers()}. 
Output layer: {self.get_output_layer()}
              """)
        
    def get_neurons(self):
        return self._neurons
    
    # Other methods

if __name__ == "__main__":
    config = np.array([2, 3, 3, 3, 1])
    network = NeuralNetwork(config)
    network.print_configuration()
    print(network.get_neurons())