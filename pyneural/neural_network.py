from layer import Layer
import numpy as np

class Network:
    
    def __init__(self, n_inputs: int, layers: np.array, 
                activation_function: str = 'sigmoid',
                random: bool = False) -> None:
        
        self._n_inputs = n_inputs
        self._n_layers = len(layers)
        self._activation_function = activation_function
        self._layers = []
        for idx, n_neurons in enumerate(layers):
            if idx == 0: # Number of inputs of the layer will be network inputs
                new_layer = Layer(n_neurons = n_neurons, 
                                  n_inputs_per_neuron = n_inputs,
                                  activation_function=activation_function,
                                  random = random)
            else: # Number of inputs of the layer will be previous layer
                new_layer = Layer(n_neurons = n_neurons, 
                                  n_inputs_per_neuron = layers[idx - 1],
                                  activation_function=activation_function,
                                  random = random)

            self._layers.append(new_layer)
                
    def __str__(self):
        return f"Neuron() with {self._n_layers} layers"
    
    def __repr__(self):
        return (f"{str(self)}\n"
                f"{self._layers}")
        
    def add_layer(self, layer: Layer) -> None:
        raise NotImplementedError
    
    def feedforward(self, input: list[float]) -> float:
        results = []
        for idx, layer in enumerate(self._layers):
            if idx == 0:
                result = layer.feedforward(input)
            else:
                result = layer.feedforward(results[idx - 1])
            results.append(result)

        if len(results[-1]) <= 1:    
            return results[-1][0]
        else:
            return results[-1]
        
    def gradient_matrix():
        ...
def mean_square_error(value: np.array, expected_value: np.array) -> float:
    return ((value - expected_value)**2).mean()
        
if __name__ == '__main__':
    neural_network = Network(n_inputs = 2, layers = [3, 3, 1])
    print(neural_network.feedforward([0, 1]))
    