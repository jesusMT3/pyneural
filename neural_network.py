import numpy as np
from neuron import Neuron

class NeuralNetwork:
    def __init__(self, input_layer: int, hidden_layers: list, output_layer: int, activation_function: str = "sigmoid"):
        self._input_layer = [0] * input_layer
        self._hidden_layers = []
        
        for i, num_neurons in enumerate(hidden_layers):
            if i == 0:
                weights = [1] * input_layer
                bias = 0
                layer = [Neuron(weights, bias, activation_function)] * num_neurons
            
            else: 
                weights = [1] * hidden_layers[i-1]
                bias = 0
                layer = [Neuron(weights, bias, activation_function)] * num_neurons

            self._hidden_layers.append(layer)
        
        
        self._output_layer = [Neuron(weights = [1] * hidden_layers[-1],
                                     bias = 0,
                                     activation_function = activation_function)
                            ] * output_layer
    
    def __repr__(self):
        return (f"NeuralNetwork(). \n"
                f"Input layer: "
                f"{self._input_layer} \n"
                f"Hidden layers: "
                f"{self._hidden_layers} \n"
                f"Output layers: "
                f"{self._output_layer}")
        
    def __str__(self):
        return "NeuralNetwork()"
    
    def set_params(self, file = None, weights: list = [], bias: float = 0):
        if weights is None and bias is None:
            raise ValueError("You must introduce one of the values")
        
        # No file method, all weights and bias will be the same
        if file is None: 
            for layer in self._hidden_layers:
                for neuron in layer:
                    if len(weights) > 1:
                        neuron.set_weights(weights)
                    else:
                        neuron.set_weights([weights] * len())
                    neuron.set_bias(bias)
                    
            for neuron in self._output_layer:
                neuron.set_weights(weights)
                neuron.set_bias(bias)
            
        # TODO: Import weights and bias through a file    
        else: 
            raise NotImplementedError
                
    def feedforward(self, x):
        if len(x) != len(self._input_layer):
            raise ValueError("Length of the input does not equal length of input layer")
        
        self._input_layer = x
        
        results = []
        for i, layer in enumerate(self._hidden_layers):
            if i == 0:
                layer_result = [neuron.feedforward(self._input_layer) for neuron in layer]
            else: 
                layer_result = [neuron.feedforward(results[i-1][j]) for j, neuron in enumerate(layer)]
                
            results.append(layer_result)
        
        self._results = results
        
        for neuron in self._output_layer:
            result = [neuron.feedforward(results[-1])]

        return result[0] if len(result) == 1 else result
        
if __name__ == "__main__":
    
    network = NeuralNetwork(2, [3, 3, 3], 1)
    input = [2, 3]
    network.set_params(weights = [1, 1, 1], bias = 0)
    print(network.feedforward(input))