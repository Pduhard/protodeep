import numpy as np


class Zeros():
    """
        Zeros initializer

        [0]
    """

    def __call__(self, shape, dtype=None):
        return np.zeros(shape)
