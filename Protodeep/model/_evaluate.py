from Protodeep.utils.format import wrap_tlist


def evaluate(self, validation_data):
    loss = 0
    eval_logs = {}
    features, targets = wrap_tlist(validation_data)

    for metric in self.metrics:
        metric.reset_state()

    pred = self.predict(features)

    for p, t in zip(pred, targets):
        loss += self.loss(p, t)

    for metric in self.metrics:
        metric.update_state(pred, targets)

    for metric in self.metrics:
        if self.val_set:
            self.logs["val_" + metric.name].append(metric.result())
        eval_logs[metric.name] = metric.result()
    if self.val_set:
        self.logs["val_loss"].append(loss)
    eval_logs["loss"] = loss
    return eval_logs
