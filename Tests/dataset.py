import numpy as np
import matplotlib.pyplot as plt
# from numba import njit

from fstat import Fstat

def time_series_split(features, targets, ssize=10):
    wfeatures = []
    wtargets = []


    for i in range(len(targets) - ssize):
        wfeatures.append(features[i:i+ssize])
        wtargets.append(targets[i+ssize][0])
    # for d in dataset[:10]:
    #     print(d)
    wfeatures = np.array(wfeatures).astype(np.float32)
    wtargets = np.array(wtargets).astype(np.float32)
    return wfeatures, wtargets


def parse_btc(file_name='BTCUSD_day.csv'):
    with open(file_name, 'r') as infile:
        lines = [line for line in infile.read().split('\n') if len(line) > 1]
        features = np.empty((len(lines), 4))
        targets = np.empty((len(lines), 1))
        lines.pop(0)
        for i, line in enumerate(lines):
            sline = line.split(",")[3:-1]
            # print(sline[-1])
            # target = sline.pop(0)
            targets[i] = float(sline[-1])
            features[i] = np.array(sline, dtype=float)
        # print(features.shape)
        # quit()
    return features, targets[:, np.newaxis]
    return time_series_split(features, targets[:, np.newaxis], ssize=10)


def parse_csv(file_name):
    fd = open(file_name, "r")
    lines = [line for line in fd.read().split() if len(line) > 1]
    features = np.empty((len(lines), 30))
    targets = np.empty((len(lines), 2))
    # print (features.shape)
    # print (targets.shape)
    # i = 0
    for i, line in enumerate(lines):
        sline = line.split(",")[1:]
        target = sline.pop(0)
        targets[i] = [1, 0] if target == "M" else [0, 1]
        features[i] = np.array(sline)
    return (features.astype(float), targets.astype(float))


def parse_mnist_csv(file_name):
    fd = open(file_name, "r")
    lines = fd.read().split()[:10000]
    _ = lines.pop(0)
    # print(header)
    # print(len(header.split(",")))
    # print(len(lines))
    # quit()
    features = np.empty((len(lines), 28, 28, 1))
    targets = np.zeros((len(lines), 10))
    # print (features.shape)
    # print (targets.shape)
    for i, line in enumerate(lines):
        sline = line.split(",")
        target = int(sline.pop(-1))
        targets[i][target] = 1
        # targets[i] = [1, 0] if target == "M" else [0, 1]
        features[i] = np.array(sline).reshape((28, 28, 1))
        # print(features[0])
        # print(targets[0])
    # plt.imshow(features[0], cmap=plt.get_cmap('gray'))
    # plt.show()
    return (features.astype(float) / 255., targets.astype(float))

# https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c


seed = 303
epsilon = 1e-8


# @njit
def _standardize(features, features_stat):

    for feature in features:
        for i, f in enumerate(feature):
            fs = features_stat[i]
            feature[i] = (f - fs.mean) / (fs.std)


class Dataset:

    test_features = None
    test_targets = None

    def split_dataset(self, test_split):
        # rng_state = np.random.get_state()
        np.random.seed(seed)
        np.random.shuffle(self.features)
        np.random.seed(seed)
        # np.random.set_state(rng_state)
        np.random.shuffle(self.targets)
        split = int((1 - test_split) * len(self.features))
        self.test_features = self.features[split:]
        self.test_targets = self.targets[split:]
        self.features = self.features[:split]
        self.targets = self.targets[:split]

    def __init__(self, file_name, test_split=0.0):
        if test_split > 1:
            test_split = 1
        elif test_split < 0:
            test_split = 0
        if 'mnist' in file_name:
            self.features, self.targets = parse_mnist_csv(file_name)
            self.split_dataset(test_split)
        elif 'BTC' in file_name:
            self.features, self.targets = parse_btc()
        else:
            self.features, self.targets = parse_csv(file_name)
            self.features_stat = [Fstat(feature) for feature in self.features.T]
            self.standardize()
            self.split_dataset(test_split)
        # plt.imshow(self.features[0], cmap=plt.get_cmap('gray'))
        # plt.show()
        # quit()

    def normalize(self):
        for feature in self.features:
            for i, f in enumerate(feature):
                fs = self.features_stat[i]
                feature[i] = (f - fs.min) / (fs.max - fs.min)

    def standardize(self):
        # _standardize(self.features, self.features_stat)
        for feature in self.features:
            for i, f in enumerate(feature):
                fs = self.features_stat[i]
                # print(fs.std)
                feature[i] = (f - fs.mean) / (fs.std)
                feature[i] = (f - fs.mean) / (fs.std + epsilon)
                # print(feature[i])
