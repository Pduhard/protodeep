import numpy as np


class GlorotUniform():
    """
        Glorot Uniform initializer

        [0 - sqrt(6 / (nin + nout))]

    """

    def __call__(self, shape, dtype=None):
        return np.random.randn(*shape) * np.sqrt(6. / (shape[1] + shape[0]))
