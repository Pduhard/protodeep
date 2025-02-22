import numpy as np

from fstat import Fstat

def time_series_split(features, targets, ssize=10):
    wfeatures = []
    wtargets = []

    for i in range(len(targets) - ssize):
        wfeatures.append(features[i:i+ssize])
        wtargets.append(targets[i+ssize][0])
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
            targets[i] = float(sline[-1])
            features[i] = np.array(sline, dtype=float)
    return features, targets[:, np.newaxis]


def parse_csv(file_name):
    fd = open(file_name, "r")
    lines = [line for line in fd.read().split() if len(line) > 1]
    features = np.empty((len(lines), 30))
    targets = np.empty((len(lines), 2))
    for i, line in enumerate(lines):
        sline = line.split(",")[1:]
        target = sline.pop(0)
        targets[i] = [1, 0] if target == "M" else [0, 1]
        features[i] = np.array(sline)
    return (features.astype(float), targets.astype(float))


def parse_mnist_csv(file_name):
    fd = open(file_name, "r")
    lines = fd.read().split()[:]
    _ = lines.pop(0)
    features = np.empty((len(lines), 28, 28, 1))
    targets = np.zeros((len(lines), 10))
    for i, line in enumerate(lines):
        sline = line.split(",")
        target = int(sline.pop(-1))
        targets[i][target] = 1
        features[i] = np.array(sline).reshape((28, 28, 1))
    return (features.astype(float) / 255., targets.astype(float))

seed = 303
epsilon = 1e-8

class Dataset:

    test_features = None
    test_targets = None

    def split_dataset(self, test_split):
        np.random.seed(seed)
        np.random.shuffle(self.features)
        np.random.seed(seed)
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
            self.features, self.targets = parse_btc(file_name)
        else:
            self.features, self.targets = parse_csv(file_name)
            self.features_stat = [Fstat(feature) for feature in self.features.T]
            self.standardize()
            self.split_dataset(test_split)

    def normalize(self):
        for feature in self.features:
            for i, f in enumerate(feature):
                fs = self.features_stat[i]
                feature[i] = (f - fs.min) / (fs.max - fs.min)

    def standardize(self):
        for feature in self.features:
            for i, f in enumerate(feature):
                fs = self.features_stat[i]
                feature[i] = (f - fs.mean) / (fs.std)
                feature[i] = (f - fs.mean) / (fs.std + epsilon)
