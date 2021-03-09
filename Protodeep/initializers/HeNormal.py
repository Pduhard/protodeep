import numpy as np


class HeNormal():

    # def __init__(self):

    def __call__(self, shape, dtype=None):
        # print("d")
        # seed 42: 0.55 val loss, in 25 epochs
        # seed 404: 0.49 val loss, in 25 epochs
        # np.random.seed(0)
        return np.random.randn(*shape) * np.sqrt(2 / shape[-1])
        # return np.zeros(shape)
