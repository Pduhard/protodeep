import numpy as np


class BinaryCrossentropy():

    def __init__(self):
        self.epsilon = 1e-8

    def __call__(self, prediction, target):
        p = prediction
        t = target
        # print((-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))).shape, target.dtype)
        # quit()
        # loss = 0
        # for i in range(p.shape[0]):
        #     loss += -np.mean(t[i] * np.log(p[i]) + (1 - t[i]) * np.log(1 - p[i]))
        # return loss
        return -np.mean(t * (np.ma.log(p)).filled(0) + (1 - t) * (np.ma.log(1 - p)).filled(0))
        # return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))

    def dr(self, prediction, target):
        # print(prediction, target)
        e = self.epsilon
        p = prediction
        t = target
        # print(type(p))
        # print(type(target))
        # care epsilon !! but this avoid divide by zero
        return -t / (p + e) + (1 - t) / (1 - p + e)
