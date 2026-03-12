import numpy as np

def sigmoid(z):
    #sigmoid function is a smooth version of a perceptron(the initially understood artificial neurons) like perceptron 
    #while tracking weights and biases will flip but sigmoid does gradual small output change if input change is small
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    # derivative of sigmoid: sigma(z) * (1 - sigma(z))
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)

def softmax(z):
    # numerically stable softmax
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)
