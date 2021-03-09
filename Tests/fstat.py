import numpy as np


class Fstat:

    def __init__(self, feature):
        self.mean = np.mean(feature)  # sum(feature) / len(feature)
        self.std = np.std(feature)  # dslr :D
        self.min = np.min(feature)
        self.max = np.max(feature)
