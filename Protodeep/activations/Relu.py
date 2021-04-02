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

    def __init__(self, max_value=None, negative_slope=0, threshold=0):
        self.max_value = max_value
        self.negative_slope = negative_slope
        self.threshold = threshold

    def __call__(self, inputs):
        return relu(inputs)

    def derivative(self, inputs):
        return relu_derivative(inputs)
