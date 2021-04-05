import numpy as np


class HeNormal():
    """
        He Normal initializer

        [0 - sqrt(2 / nout)]
    """

    def __call__(self, shape, dtype=None):
        return np.random.randn(*shape) * np.sqrt(2 / shape[-1])
