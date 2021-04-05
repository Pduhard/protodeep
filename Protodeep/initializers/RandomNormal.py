import numpy as np


class RandomNormal():
    """
        Random Normal initializer

        [0 - 1]
    """
    def __call__(self, shape, mean=0.0, stddev=0.05):
        # print("d")
        return np.random.normal(loc=mean, scale=stddev, size=shape)
        # return np.zeros(shape)
