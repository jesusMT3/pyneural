import numpy as np
from scipy.special import erf
from math import sqrt

"""
In this module it is developed all of the most common
activation functions used for creating neurons on
neural networks. The source is from wikipedia:
https://en.wikipedia.org/wiki/Activation_function
"""

def identity(x: float) -> float:
    """
    Identity activation function: \\
    f(x) = x
    """
    return x

def binaryStep(x: float) -> float:
    """
    Binary step activation function: \\
    f(x) = 0 if x < 0 \\
    f(x) = 1 if x >= 0 
    """
    return 0 if x < 0 else 1

    
def sigmoid(x: float) -> float:
    """
    Sigmoid activation function: \\
    f(x) = 1/(1 + e^(-x))
    """
    return 1/((1 + np.exp(-x)))

def tanh(x: float) -> float:
    """
    Hyperbolic tangent activation function: \\
    f(x) = (e^x - e^-x)/(e^x + e^-x)
    """
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def smht(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Soboleva modified hyperbolic tangent activation function: \\
    f(x) = (e^ax - e^-bx)/(e^cx + e^-dx)
    """
    return (np.exp(a * x) - np.exp(b * (-x)))/(np.exp(c * x) + np.exp(d * (-x)))

def reLu(x: float) -> float:
    """
    Recified linear unit activation function: \\
    f(x) = 0 if x < 0 \\
    f(x) = x if x >= 0 
    """
    return 0 if x < 0 else x

def geLu(x: float) -> float:
    """
    Gaussian error linear unit activation function: \\
    f(x) = 0.5x * (1 + erf(x/sqrt(2)))
    """
    return 0.5 * x * (1 + erf(x/sqrt(2)))

def softPlus(x: float) -> float:
    """
    Softplus activation function: \\
    f(x) = ln(1+e^x)
    """
    return np.log(1+np.exp(x))

def eLu(x: float, alpha: float) -> float:
    """
    Exponential linear unit activation function: \\
    f(x) = alpha(e^x - 1) if x < 0
    f(x) = x if x > 0
    """
    return alpha * (np.exp(x) - 1) if x < 0 else x

def seLu(x: float):
    """
    Scaled exponential linear unit activation function \\
    f(x) = lambda * alpha(e^x - 1) if x < 0
    f(x) = lambda * x if x >= 0
    with parameters: 
    lambda = 1.0507
    alpha = 1.67326
    """
    return 1.67326 * 1.0507 * (np.exp(x) - 1) if x < 0 else 1.0507 * x

def leakyReLu(x: float):
    """
    Leaky rectified linear unit activation function \\
    f(x) = x if > 0
    f(x) = 0.01x if x <= 0
    """
    return x if x > 0 else 0.01 * x

def preLu(x: float, alpha: float):
    """
    Parametric rectified linear unit activation function \\
    f(x) = alpha * x if x < 0
    f(x) = x if x >= 0
    """
    return alpha * x if x < 0 else x

def siLu(x: float):
    """
    Sigmoid linear unit activation function
    f(x) = x / (1 + e^(-x))
    """
    return x / (1 + np.exp(-x))

def gaussian(x: float):
    """
    Gaussian activation function
    f(x) = e^(-x^2)
    """
    return np.exp(-x ** 2)

# Vectorial activation functions
def softmax(x: np.array):
    output = []
    for i in x:
        num = np.exp(i)
        den = 0
        for j in x:
            den += np.exp(j)
        
        output.append(num/den)
    
    return output

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