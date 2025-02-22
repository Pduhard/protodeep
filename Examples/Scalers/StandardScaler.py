from Scalers.Scaler import Scaler
from stats import Stats
import numpy as np
import json


class StandardScaler(Scaler):

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, dataset):
        # self.mean = np.mean(dataset, axis=0)
        # self.std = np.std(dataset, axis=0)
        self.mean = Stats.mean(dataset)
        self.std = Stats.std(dataset, mean=self.mean)
        return self

    def transform(self, dataset):
        if len(dataset.shape) < 2:
            dataset = dataset[:, np.newaxis]
        return (dataset - self.mean) / self.std

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

    def save(self, file_name='scaler.json'):
        with open(file_name, 'w+') as outfile:
            json.dump({
                'class': 'StandardScaler',
                'mean': self.mean.tolist(),
                'std': self.std.tolist()
            }, outfile)
        return self

    def load(self, file_name='scaler.json'):
        with open(file_name, 'r+') as outfile:
            dataset = json.load(outfile)
            self.mean = np.array(dataset['mean'])
            self.std = np.array(dataset['std'])
        return self
