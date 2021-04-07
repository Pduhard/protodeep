import numpy as np


class L1():

    def __init__(self, l1=0.01):
        self.l1 = l1

    def __call__(self, inputs):
        return self.l1 * np.sum(np.abs(inputs))

    def derivative(self, inputs):
        return self.l1 * np.where(inputs < 0, -1., 1.)
