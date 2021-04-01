import numpy as np


class L2():

    def __init__(self, l2=0.01):
        self.l2 = l2

    def __call__(self, inputs):
        return self.l2 * np.sum(np.square(inputs))

    def derivative(self, inputs):
        return self.l2 * inputs * 2
