import numpy as np


class MeanSquaredError():

    def __init__(self):
        self.epsilon = 1e-8

    def __call__(self, prediction, target):
        delta = prediction - target
        return np.mean(np.mean(delta * delta, axis=0))

    def dr(self, prediction, target):
        return (prediction - target) * 2
