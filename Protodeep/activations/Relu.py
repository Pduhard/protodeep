from Protodeep.activations.Activation import Activation
import numpy as np
from numba import njit


@njit
def relu(inputs):
    return np.where(inputs < 0, 0., inputs)


@njit
def relu_derivative(inputs):
    return np.where(inputs > 0, 1., 0.)


class Relu(Activation):
    """
        ReLu activation:
            f(x) = x if x > 0 else 0
            f'(x) = 1 if x > 0 else 0
    """

    def __init__(self):
        pass

    def __call__(self, inputs):
        return relu(inputs)

    def derivative(self, inputs):
        return relu_derivative(inputs)
