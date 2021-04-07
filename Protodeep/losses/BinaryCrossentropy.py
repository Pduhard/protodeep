import numpy as np
from Protodeep.utils.debug import class_timer


@class_timer
class BinaryCrossentropy():

    def __init__(self):
        self.epsilon = 1e-8

    def __call__(self, prediction, target):
        e = self.epsilon
        p = prediction
        t = target
        return -np.mean(t * np.log(p + e) + (1 - t) * np.log(1 - p + e))

    def dr(self, prediction, target):
        e = self.epsilon
        p = prediction
        t = target
        return -t / (p + e) + (1 - t) / (1 - p + e)
