from pyneural.activation_functions import *

datapoints = [-0.5, 0, 0.5]

def test_identity():
    assert identity(datapoints[0]) == -0.5, "Result should be -0.5"
    assert identity(datapoints[1]) == 0, "Result should be 0"
    assert identity(datapoints[2]) == 0.5, "Result should be 0.5"
