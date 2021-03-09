import numpy as np


class Zeros():

    # def __init__(self):

    def __call__(self, shape, dtype=None):
        # print(shape[1])
        # print("d")
        # print(np.random.rand(*shape).shape)
        return np.zeros(shape)
