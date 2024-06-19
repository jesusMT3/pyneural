from neuron import Neuron
import numpy as np

class Layer():
    def __init__(self,n_neurons, n_inputs_per_neuron, 
                 activation_function = "sigmoid", random: bool = False):
        
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
        print(self._neurons)

    # Add one neuron of same type or one particular neuron
    def add_neuron(self, neuron: Neuron = None, random: bool = False):
        if neuron is None:
            self._neurons = np.append(self._neurons, 
                                  Neuron(weights = np.ones(self._n_weights),
                                        bias = 0,
                                        activation_function = self._activation_function))
        
        else: 
            self._neurons = np.append(self._neurons, neuron)
    def add_neurons(self, neurons_to_append: int):
        ...
        
    def feedforward(self, inputs: np.array):
        output = np.zeros(len(self._neurons))

        # Run through the neurons
        for idx, neuron in enumerate(self._neurons):
            output[idx] = neuron.feedforward(inputs)

        return output

class Neural_Network():
    def __init(self):
        ...

layer = Layer(n_neurons = 3, n_inputs_per_neuron = 3, random = False)
print(layer.feedforward([1] * 3))