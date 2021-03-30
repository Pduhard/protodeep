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
# def vtanh(x, xsum):
#     return np.exp(x) / (xsum + epsilon)

@njit
def tanh(inputs):
    return 2 / (1 + np.exp(-2 * inputs)) - 1
    # while (np.max(np.abs(inputs)) > 10):
    #     inputs /= 2
    # xsum = np.sum(np.exp(inputs))
    # return np.array([np.exp(x) / (xsum + epsilon) for x in inputs])
    # return (np.vectorize(vtanh)(inputs, xsum))


@njit
def tanh_derivative(inputs):
    activ = tanh(inputs)
    return 1 - activ * activ


class Tanh(Activation):

    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, inputs):
        return tanh(inputs)

    def derivative(self, inputs):
        return tanh_derivative(inputs)
