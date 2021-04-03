from Protodeep.activations.Activation import Activation
import numpy as np
from numba import njit


@njit(fastmath=True)
def softmax(inputs):
    sft = np.empty(inputs.shape)
    for i in range(inputs.shape[0]):
        xsum = np.sum(np.exp(inputs[i]))
        sft[i, ...] = np.exp(inputs[i, ...]) / xsum
    return sft


@njit(fastmath=True)
def softmax_derivative(inputs):
    sft = np.empty(inputs.shape)
    for i in range(inputs.shape[0]):
        xsum = np.sum(np.exp(inputs[i]))
        sft[i, ...] = np.exp(inputs[i, ...]) / xsum
    return sft * (1. - sft)


class Softmax(Activation):
    """
        Sigmoid activation:
            f(x) = 1 / (1 + exp(-x))
            f'(x) = f(x) * (1 - f(x))
    """
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, inputs):
        return softmax(inputs)

    def derivative(self, inputs):
        return softmax_derivative(inputs)
