import numpy as np
from Protodeep.utils.format import wrap_tlist


def evaluate(self, validation_data):
    loss = 0

    features, targets = wrap_tlist(validation_data)
    test_size = len(features[0])

    for metric in self.metrics:
        metric.reset_state()

    # for i in range(test_size):
    #     feature = [f[i] for f in features]
    #     target = [t[i] for t in targets]
    # print(test_size)
    pred = self.predict(features)

    for p, t in zip(pred, targets):
        # print(p.shape)
        # print(t.shape)
        loss += self.loss(p, t)

    for metric in self.metrics:
        metric.update_state(pred, targets)

    for metric in self.metrics:
        self.logs["val_" + metric.name].append(metric.result())
    self.logs["val_loss"].append(loss)
