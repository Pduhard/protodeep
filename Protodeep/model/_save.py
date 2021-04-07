import json
import numpy as np


def save_weights(self, file_name='model_weights.json'):
    with open(file_name, 'w+') as outfile:
        dc = {
            layer.name: [
                weight.tolist() for weight in layer.get_weights()
            ] for layer in self.flatten_graph if layer.trainable
        }
        json.dump(dc, outfile)
    return self


def load_weights(self, file_name='model_weights.json'):

    with open(file_name, 'r+') as outfile:
        dataset = json.load(outfile)
        for layer in self.flatten_graph:
            if layer.name in dataset:
                layer.set_weights(
                    [np.array(weight) for weight in dataset[layer.name]]
                )
    self.weights = []
    self.gradients = []
    self.build()
    return self


def get_weights(self):
    return self.weights


def set_weights(self, new_weights):
    if len(self.weights) != len(new_weights):
        print(
            'set weight error: New weights does not contain'
            'as many matrix as expected'
            f'expected: {len(self.weights)} matrix'
            f'actual: {len(new_weights)} matrix'
        )
    for weight, new_weight in zip(self.weights, new_weights):
        weight.fill(0)
        weight += new_weight
