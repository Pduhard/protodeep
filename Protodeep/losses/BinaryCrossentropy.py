import numpy as np


class BinaryCrossentropy():

    def __init__(self):
        self.epsilon = 1e-8

    def __call__(self, prediction, target):
        p = prediction
        t = target
        return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))

    def dr(self, prediction, target):
        # print(prediction, target)
        e = self.epsilon
        p = prediction
        t = target
        # print(type(p))
        # print(type(target))
        # care epsilon !! but this avoid divide by zero
        return -t / (p + e) + (1 - t) / (1 - p + e)
