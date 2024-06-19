from pyneural import activation_functions as af

datapoints = [-0.5, 0, 0.5]

def test_identity():
    assert af.identity(datapoints[0]) == -0.5
    
