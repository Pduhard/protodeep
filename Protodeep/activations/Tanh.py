from Protodeep.activations.Activation import Activation
import numpy as np
from numba import njit


@njit(fastmath=True)
def tanh(inputs):
    return 2. / (1. + np.exp(-2. * inputs)) - 1.


@njit(fastmath=True)
def tanh_derivative(inputs):
    activ = 2. / (1. + np.exp(-2. * inputs)) - 1.
    return 1 - activ * activ


class Tanh(Activation):
    """
        Tanh activation:
            f(x) = 2 / (1 + exp(-2x)) - 1
            f'(x) = 1 - f(x)²
    """
    def __init__(self):
        pass

    def __call__(self, inputs):
        return tanh(inputs)

    def derivative(self, inputs):
        return tanh_derivative(inputs)
