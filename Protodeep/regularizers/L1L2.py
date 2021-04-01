import numpy as np


class L1L2():

    def __init__(self, l1=0.01, l2=0.01):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, inputs):
        return self.l1 * np.sum(np.abs(inputs)) + self.l2 * np.sum(np.square(inputs))

    def derivative(self, inputs):
        return self.l1 * np.where(inputs < 0, -1., 1.) + self.l2 * inputs * 2.
