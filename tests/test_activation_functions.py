import pyneural.activation_functions as af

datapoints = [-0.5, 0, 0.5]

def test_identity():
    assert af.identity(datapoints[0]) == -0.5, "Result should be -0.5"
    assert af.identity(datapoints[1]) == 0, "Result should be 0"
    assert af.identity(datapoints[2]) == 0.5, "Result should be 0.5"

if __name__ == "__main__":
    test_identity()