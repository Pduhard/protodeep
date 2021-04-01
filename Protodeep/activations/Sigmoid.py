from Protodeep.activations.Activation import Activation
import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(func):
        return func

epsilon = 1e-8


# @jit
# @vectorize


# @jit
# def vsigmoid(x, xsum):
#     return np.exp(x) / (xsum + epsilon)

# @njit
def sigmoid(inputs):
    return 1 / (1 + np.exp(-inputs))
    # while (np.max(np.abs(inputs)) > 10):
    #     inputs /= 2
    # xsum = np.sum(np.exp(inputs))
    # return np.array([np.exp(x) / (xsum + epsilon) for x in inputs])
    # return (np.vectorize(vsigmoid)(inputs, xsum))


# @njit
def sigmoid_derivative(inputs):
    activ = sigmoid(inputs)
    return activ * (1 - activ)


class Sigmoid(Activation):

    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, inputs):
        return sigmoid(inputs)

    def derivative(self, inputs):
        return sigmoid_derivative(inputs)
