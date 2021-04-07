import numpy as np


def get_mini_batchs(self, features, targets, batch_size):
    rng_state = np.random.get_state()
    for f in features:
        np.random.shuffle(f)
        np.random.set_state(rng_state)
    for t in targets:
        np.random.shuffle(t)
        np.random.set_state(rng_state)
    dataset_size = len(features[0])
    section_size_rest = dataset_size % batch_size
    section_nb = (dataset_size - section_size_rest) // batch_size
    rest = 1 if (dataset_size - section_size_rest) % batch_size != 0 else 0
    section = [batch_size * i for i in range(1, section_nb + rest)]
    return (
        [np.split(f, section) for f in features],
        [np.split(t, section) for t in targets],
        section_nb + rest
    )


def init_logs(self):
    self.logs = {}
    for metric in self.metrics:
        if self.val_set:
            self.logs["val_" + metric.name] = []
        self.logs[metric.name] = []
    if self.val_set:
        self.logs["val_loss"] = []
    self.logs["loss"] = []


def update_metrics(self, loss):
    for metric in self.metrics:
        self.logs[metric.name].append(metric.result())
    self.logs["loss"].append(loss)


def reset_gradients(self):
    for layer in self.layers:
        layer.reset_gradients()
