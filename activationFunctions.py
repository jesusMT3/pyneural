import numpy as np
from scipy.special import erf
from math import sqrt

def identity(x: float) -> float:
    """Identity activation function: \\
    f(x) = x
    """
    return x

def binaryStep(x: float) -> float:
    """Binary step activation function: \\
    f(x) = 0 if x < 0 \\
    f(x) = 1 if x >= 0 
    """
    return 0 if x < 0 else 1

    
def sigmoid(x: float) -> float:
    """Sigmoid activation function: \\
    f(x) = 1/(1 + e^(-x))
    """
    return 1/((1 + np.exp(-x)))

def tanh(x: float) -> float:
    """Hyperbolic tangent activation function: \\
    f(x) = (e^x - e^-x)/(e^x + e^-x)
    """
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def smht(x: float, a: float = 1, b: float = 1, c: float = 1, d: float = 1) -> float:
    """Soboleva modified hyperbolic tangent activation function: \\
    f(x) = (e^ax - e^-bx)/(e^cx + e^-dx)
    """
    return (np.exp(a * x) - np.exp(b * (-x)))/(np.exp(c * x) + np.exp(d * (-x)))

def reLu(x: float) -> float:
    """Recified linear unit activation function: \\
    f(x) = 0 if x < 0 \\
    f(x) = x if x >= 0 
    """
    return 0 if x < 0 else x

def geLu(x: float) -> float:
    """Gaussian error linear unit activation function: \\
    f(x) = 0.5x * (1 + erf(x/sqrt(2)))
    """
    return 0.5 * x * (1 + erf(x/sqrt(2)))

def softPlus(x: float) -> float:
    """Softplus activation function: \\
    f(x) = ln(1+e^x)
    """
    return np.log(1+np.exp(x))

def eLu(x: float, alpha: float = 1) -> float:
    """Exponential linear unit activation function: \\
    f(x) = alpha(e^x - 1) if x < 0
    f(x) = x if x > 0
    """
    return alpha * (np.exp(x) - 1) if x < 0 else x

def seLu():
    raise(NotImplementedError)

def leakyReLu():
    raise(NotImplementedError)

def preLu():
    raise(NotImplementedError)

def siLu():
    raise(NotImplementedError)

def gaussian():
    raise(NotImplementedError)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = [round(i * 0.1, 1) for i in range(-100, 101)]
    y = [eLu(i) for i in x]
    plt.plot(x, y)
    plt.show()