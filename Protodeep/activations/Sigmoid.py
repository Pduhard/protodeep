from Protodeep.activations.Activation import Activation
import numpy as np
from numba import njit


@njit(fastmath=True)
def sigmoid(inputs):
    return 1 / (1 + np.exp(-inputs))


@njit(fastmath=True)
def sigmoid_derivative(inputs):
    activ = 1 / (1 + np.exp(-inputs))
    return activ * (1 - activ)


class Sigmoid(Activation):
    """
        Sigmoid activation:
            f(x) = 1 / (1 + exp(-x))
            f'(x) = f(x) * (1 - f(x))
    """

    def __init__(self):
        pass

    def __call__(self, inputs):
        return sigmoid(inputs)

    def derivative(self, inputs):
        return sigmoid_derivative(inputs)
