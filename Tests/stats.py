import numpy as np


class Stats:

    @staticmethod
    def mean(dataset):
        mean = []
        """ works like np.mean(dataset, axis=0) """
        if len(dataset.shape) < 2:
            dataset = dataset[:, np.newaxis]
        for i in range(dataset.shape[-1]):
            fsum = 0
            for feature in dataset:
                fsum += feature[i]
            mean.append(fsum / dataset.shape[0])
        return np.array(mean)

    @staticmethod
    def var(dataset, mean=None):
        """ works like np.var(dataset, axis=0) """
        if mean is None:
            mean = Stats.mean(dataset)
        var = []
        if len(dataset.shape) < 2:
            dataset = dataset[:, np.newaxis]
        for i in range(dataset.shape[-1]):
            fvarsum = 0
            for feature in dataset:
                fvarsum += (feature[i] - mean[i]) ** 2
            var.append(fvarsum / dataset.shape[0])
        return np.array(var)

    @staticmethod
    def std(dataset, mean=None, var=None):
        """ works like np.std(dataset, axis=0) """
        if len(dataset.shape) < 2:
            dataset = dataset[:, np.newaxis]
        if mean is None:
            mean = Stats.mean(dataset)
        if var is None:
            var = Stats.var(dataset, mean=mean)
        return np.sqrt(var)

    @staticmethod
    def min(dataset):
        """ works like np.min(dataset, axis=0) """
        minv = []
        if len(dataset.shape) < 2:
            dataset = dataset[:, np.newaxis]
        for i in range(dataset.shape[-1]):
            bestmin = np.inf
            for feature in dataset:
                if feature[i] < bestmin:
                    bestmin = feature[i]
            minv.append(bestmin)
        return np.array(minv)

    @staticmethod
    def max(dataset):
        """ works like np.max(dataset, axis=0) """
        maxv = []
        if len(dataset.shape) < 2:
            dataset = dataset[:, np.newaxis]
        for i in range(dataset.shape[-1]):
            bestmax = -np.inf
            for feature in dataset:
                if feature[i] > bestmax:
                    bestmax = feature[i]
            maxv.append(bestmax)
        return np.array(maxv)
