import numpy as np
from Protodeep.utils.format import cwrap_list, wrap_tlist
from Protodeep.utils.debug import class_timer

epsilon = 1e-8


@class_timer
class Model:

    from ._build import add, compile, build_graph, build
    from ._evaluate import evaluate
    from ._print import summary, layer_summary, print_epoch_metrics
    from ._model_utils import get_mini_batchs, init_logs, update_metrics
    from ._save import save_weights, load_weights, get_weights, set_weights

    def __init__(self, inputs=None, outputs=None):
        self.layers = []
        self.weights = []
        self.gradients = []
        self.metrics = []
        self.optimizer = None
        self.loss = None
        self.logs = None
        self.val_set = False
        self.graph = {}
        self.flatten_graph = None
        self.linked = False
        self.linear = True
        if inputs is not None and outputs is not None:
            self.linked = True
            self.build_graph(inputs, outputs)

    def backpropagate(self, dloss):
        for layer in self.flatten_graph[::-1]:
            inputs = None
            if layer in self.outputs:
                inputs = np.array(dloss[self.outputs.index(layer)])
            for next_l in layer.output_connectors.next_layers:
                if inputs is None:
                    inputs = np.array(next_l.dloss)
                else:
                    inputs += next_l.dloss
            layer.backward_pass(inputs)

    def predict(self, feature):
        if not isinstance(feature, list):
            feature = [feature]
        for layer in self.flatten_graph:
            if layer in self.inputs:
                inputs = np.array(feature[self.inputs.index(layer)])
            else:
                inputs = layer.input_connectors.layer.a_val
            layer.forward_pass(inputs)
        return [o.a_val for o in self.outputs]

    def fit_step(self, features, targets):
        pred = self.predict(features)
        dp_loss = []
        loss = 0
        for p, t in zip(pred, targets):
            loss += self.loss(p, t)
            dp_loss.append(self.loss.dr(p, t))
        for metric in self.metrics:
            metric.update_state(pred, targets)
        self.backpropagate(dp_loss)
        self.optimizer.apply_gradient(self.weights, self.gradients)
        return loss

    def init_fit(self, features, targets, validation_data,
                 callbacks, batch_size):
        features = cwrap_list(features)
        targets = cwrap_list(targets)
        self.val_set = validation_data is not None
        self.init_logs()
        if self.val_set:
            validation_data = wrap_tlist(validation_data)
        if callbacks is not None:
            for c in callbacks:
                c.set_model(self)
                c.on_fit_start()
        if batch_size > len(features[0]):
            batch_size = len(features[0])
        return features, targets, validation_data, batch_size

    def fit(self, features, targets, epochs=10, batch_size=32,
            validation_data=None, callbacks=None, verbose=True):
        features, targets = cwrap_list(features), cwrap_list(targets)
        self.val_set = validation_data is not None
        self.init_logs()
        if self.val_set:
            validation_data = wrap_tlist(validation_data)
        if callbacks is not None:
            for c in callbacks:
                c.set_model(self)
                c.on_fit_start()
        if batch_size > len(features[0]):
            batch_size = len(features[0])
        for e in range(epochs):
            for metric in self.metrics:
                metric.reset_state()
            loss = 0
            bx, by, steps = self.get_mini_batchs(features, targets, batch_size)
            for s in range(steps):
                loss += self.fit_step([x[s] for x in bx], [y[s] for y in by])
            loss /= steps
            self.update_metrics(loss)
            if self.val_set:
                self.evaluate(validation_data)
            if verbose:
                self.print_epoch_metrics(e, epochs)
            if callbacks is not None:
                for callback in callbacks:
                    if callback.on_epoch_end(self.logs) is False:
                        return self.logs
        return self.logs
