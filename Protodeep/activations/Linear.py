from Protodeep.activations.Activation import Activation
import numpy as np
# try:
#     from numba import njit
# except ImportError:
#     def njit(func):
#         return func

class Linear(Activation):

    def __init__(self):
        pass

    def __call__(self, inputs):
        return np.array(inputs)

    def derivative(self, inputs):
        return np.ones(inputs.shape)
