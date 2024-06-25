from neuron import Neuron
import numpy as np

class Layer():
    def __init__(self,n_neurons, n_inputs_per_neuron, 
                 activation_function = "sigmoid", random: bool = False) -> None:
        
        self._n_weights = n_inputs_per_neuron
        self._n_neurons = n_neurons
        self._activation_function = activation_function
        
        # Set random parameters for the weights and bias of the neuron
        if random == True:
            neurons = []
            for _ in range(n_neurons):
                weights = np.random.rand(self._n_weights)
                bias = np.random.rand()
                neuron = Neuron(weights, bias, activation_function)
                neurons.append(neuron)
                
            self._neurons = np.array(neurons, dtype = Neuron)

        # Default all weights to 1 and all bias to 0
        else:
            self._neurons = np.array([Neuron(weights = np.ones(self._n_weights),
                                            bias = 0,
                                            activation_function = activation_function)] 
                                    * n_neurons, dtype = Neuron)

    def __str__(self) -> str:
        return f"Layer() with {self._n_neurons} neurons"
    
    def __repr__(self) -> str:
        return (f"{str(self)}\n"
                f"{self._neurons}")

    # Add one neuron of same type or one particular neuron
    def add_neuron(self, neuron: Neuron = None) -> None:
        if neuron is None:
            self._neurons = np.append(self._neurons, 
                                  Neuron(weights = np.ones(self._n_weights),
                                        bias = 0,
                                        activation_function = self._activation_function))
        
        else: 
            self._neurons = np.append(self._neurons, neuron)

    def add_neurons(self, neurons_to_append: int):
        raise NotImplementedError
        
    def feedforward(self, inputs: np.array):
        output = np.zeros(len(self._neurons))

        # Run through the neurons
        for idx, neuron in enumerate(self._neurons):
            output[idx] = neuron.feedforward(inputs)

        return output

if __name__ == "__main__":

    layer = Layer(n_neurons = 4, n_inputs_per_neuron = 3, random = True)

    print(repr(layer))

    neuron = Neuron(weights = np.zeros(layer._n_weights), bias = 1, activation_function="relu")
    layer.add_neuron(neuron)
    print(repr(layer))