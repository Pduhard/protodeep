from Protodeep.activations.Activation import Activation
import numpy as np


class Linear(Activation):
    """
        Linear activation:
            f(x) = x
            f'(x) = 1
    """
    def __init__(self):
        pass

    def __call__(self, inputs):
        return np.array(inputs)

    def derivative(self, inputs):
        return np.ones(inputs.shape)
