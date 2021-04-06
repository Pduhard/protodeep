import numpy as np
from Protodeep.utils.debug import class_timer


@class_timer
class BinaryCrossentropy():

    def __init__(self):
        self.epsilon = 1e-8

    def __call__(self, prediction, target):
        p = prediction
        t = target
        return -np.mean(np.where(t == 0, np.log(1 - p) if p != 1 else 0, np.log(p) if p != 0 else 0))
        # return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))
        # return -np.mean(t * (np.ma.log(p)).filled(0) + (1 - t) * (np.ma.log(1 - p)).filled(0))

    def dr(self, prediction, target):
        e = self.epsilon
        p = prediction
        t = target
        return -t / (p + e) + (1 - t) / (1 - p + e)
