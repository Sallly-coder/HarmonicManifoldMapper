import numpy as np
#Cost function punishes errors, mistakes made by the weights and biases transcending through 
#layers by providing how wrong is it in detecting happy or sad
def quadratic_cost(a_output, y_true):
    """
    C = (1/2n) * sum of squared differences
    a_output: network output  shape (num_classes, num_samples)
    y_true:   correct labels  shape (num_classes, num_samples)
    """
    n = y_true.shape[1]
    return (1 / (2 * n)) * np.sum((a_output - y_true) ** 2)

def quadratic_cost_derivative(a_output, y_true):
    """
    del C/del a at output layer = (a - y)
    This feeds directly into BP1
    """
    return (a_output - y_true)
