#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
In this module it is developed all of the most common
activation functions used for creating neurons on
neural networks. The source is from wikipedia:
https://en.wikipedia.org/wiki/Activation_function

Here is the complete list of the implemented activation functions and their representations:
    - Identity: "identity"
    - Binary step: "binary_step"
    - Sigmoid: "sigmoid"
    - Hyperbolic tangent: "tanh"
    - Soboleva modified hyperbolic tangent: "smht"
    - Recified linear unit: "relu"
    - Gaussian error linear unit: "gelu"
    - Soft-plus: "soft_plus"
    - Exponential linear unit: "elu"
    - Scaled exponential linear unit: "selu"
    - Leaky rectified linear unit: "leaky_relu"
    - Parametric rectified linear unit: "prelu"
    - Sigmoid linear unit: "silu"
    - Gaussian: "gaussian"
    
Some vectorial activation functions are implemented as well:
    - Soft Max: "softmax"
"""

import numpy as np
from scipy.special import erf
from math import sqrt

def identity(input_value: float) -> float:
    """
    Identity activation function: \\
    f(x) = x
    """
    return input_value

def binary_step(input_value: float) -> float:
    """
    Binary step activation function: \\
    f(x) = 0 if x < 0 \\
    f(x) = 1 if x >= 0 
    """
    return 0 if input_value < 0 else 1


def sigmoid(input_value: float) -> float:
    """
    Sigmoid activation function: \\
    f(x) = 1/(1 + e^(-x))
    """
    return 1/((1 + np.exp(-input_value)))

def tanh(input_value: float) -> float:
    """
    Hyperbolic tangent activation function: \\
    f(x) = (e^x - e^-x)/(e^x + e^-x)
    """
    num = np.exp(input_value) - np.exp(-input_value)
    den = np.exp(input_value) + np.exp(-input_value)
    return num / den

def smht(input_value: float,
        param_a: float,
        param_b: float,
        param_c: float,
        param_d: float) -> float:
    """
    Soboleva modified hyperbolic tangent activation function: \\
    f(x) = (e^ax - e^-bx)/(e^cx + e^-dx)
    """
    num = np.exp(param_a * input_value) - np.exp(param_b * (-input_value))
    den = np.exp(param_c * input_value) + np.exp(param_d * (-input_value))
    return num / den

def relu(input_value: float) -> float:
    """
    Recified linear unit activation function: \\
    f(x) = 0 if x < 0 \\
    f(x) = x if x >= 0 
    """
    return 0 if input_value < 0 else input_value

def gelu(input_value: float) -> float:
    """
    Gaussian error linear unit activation function: \\
    f(x) = 0.5x * (1 + erf(x/sqrt(2)))
    """
    return 0.5 * input_value * (1 + erf(input_value/sqrt(2)))

def soft_plus(input_value: float) -> float:
    """
    Softplus activation function: \\
    f(x) = ln(1+e^x)
    """
    return np.log(1 + np.exp(input_value))

def elu(input_value: float, alpha: float) -> float:
    """
    Exponential linear unit activation function: \\
    f(x) = alpha(e^x - 1) if x < 0
    f(x) = x if x > 0
    """
    return alpha * (np.exp(input_value) - 1) if input_value < 0 else input_value

def selu(input_value: float):
    """
    Scaled exponential linear unit activation function \\
    f(x) = lambda * alpha(e^x - 1) if x < 0
    f(x) = lambda * x if x >= 0
    with parameters: 
    lambda = 1.0507
    alpha = 1.67326
    """
    return 1.67326 * 1.0507 * (np.exp(x) - 1) if input_value < 0 else 1.0507 * input_value

def leaky_relu(input_value: float):
    """
    Leaky rectified linear unit activation function \\
    f(x) = x if > 0
    f(x) = 0.01x if x <= 0
    """
    return input_value if input_value > 0 else 0.01 * input_value

def prelu(input_value: float, alpha: float):
    """
    Parametric rectified linear unit activation function \\
    f(x) = alpha * x if x < 0
    f(x) = x if x >= 0
    """
    return alpha * input_value if input_value < 0 else input_value

def silu(input_value: float):
    """
    Sigmoid linear unit activation function
    f(x) = x / (1 + e^(-x))
    """
    return input_value / (1 + np.exp(-input_value))

def gaussian(input_value: float):
    """
    Gaussian activation function
    f(x) = e^(-x^2)
    """
    return np.exp(-input_value ** 2)

# Vectorial activation functions
def softmax(input_array: np.array):
    """
    Softmax activation function
    """
    output_array = []
    for i in input_array:
        num = np.exp(i)
        den = 0
        for j in x:
            den += np.exp(j)

        output_array.append(num/den)

    return output_array 

if __name__ == "__main__":
    # activation functions test
    import matplotlib.pyplot as plt
    x = [round(i * 0.1, 1) for i in range(-100, 101)]
    y = [gaussian(i) for i in x]
    plt.plot(x, y)
    plt.show()

    # softmax function test
    import random
    x = np.random.rand(10) * 10
    print(x, "\n")
    y = softmax(x)
    print(y, "\n")
    print(sum(y))
