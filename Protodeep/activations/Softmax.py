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
# def vsoftmax(x, xsum):
#     return np.exp(x) / (xsum + epsilon)

@njit
def softmax(inputs):
    while (np.max(np.abs(inputs)) > 10):
        inputs /= 2
    xsum = np.sum(np.exp(inputs))
    return np.array([np.exp(x) / (xsum + epsilon) for x in inputs])
    # return (np.vectorize(vsoftmax)(inputs, xsum))


@njit
def softmax_derivative(inputs):
    activ = softmax(inputs)
    return activ * (1 - activ)


class Softmax(Activation):

    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, inputs):
        return softmax(inputs)

    def derivative(self, inputs):
        return softmax_derivative(inputs)