import numpy as np


class GlorotNormal():
    """
        Glorot Normal initializer AKA Xavier initializer

        [0 - sqrt(2 / (nin + nout))]

    """

    def __call__(self, shape, dtype=None):
        return np.random.randn(*shape) * np.sqrt(2. / (shape[1] + shape[0]))
