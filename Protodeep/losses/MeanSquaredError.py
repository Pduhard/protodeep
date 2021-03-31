import numpy as np


class MeanSquaredError():

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
        d = p - t
        # print(p.shape)
        # print(t.shape)
        # quit()
        return np.mean(np.mean(d * d, axis=0))

    def dr(self, prediction, target):
        # print(prediction, target)
        e = self.epsilon
        p = prediction
        t = target
        # print(type(p))
        # print(type(target))
        return (p - t) * 2
