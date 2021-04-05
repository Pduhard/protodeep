import numpy as np
from Protodeep.utils.format import wrap_tlist


def evaluate(self, validation_data):
    loss = 0
    eval_logs = {}
    features, targets = wrap_tlist(validation_data)

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
        eval_logs[metric.name] = metric.result()
    self.logs["val_loss"].append(loss)
    eval_logs["loss"] = loss
    return eval_logs


