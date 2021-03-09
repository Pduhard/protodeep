import numpy as np


class GlorotNormal():

    # def __init__(self):

    def __call__(self, shape, dtype=None):
        # print(shape[0])
        # print(shape[1])
        # quit()
        return np.random.randn(*shape) * np.sqrt(2 / (shape[1] + shape[0]))
        # return np.zeros(shape)
