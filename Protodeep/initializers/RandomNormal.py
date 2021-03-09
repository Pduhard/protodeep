import numpy as np


class RandomNormal():

    # def __init__(self):

    def __call__(self, shape, mean=0.0, stddev=0.05):
        # print("d")
        return np.random.normal(loc=mean, scale=stddev, size=shape)
        # return np.zeros(shape)
