import numpy as np
import json
# from numba import njit
# from numba import jit, vectorize
# from activations.Relu import Relu
# from activations.Softmax import Softmax
from Protodeep.utils.format import cwrap_list, wrap_tlist
from Protodeep.utils.debug import class_timer

epsilon = 1e-8


@class_timer
class Model:

    from ._build import add, compile, build_graph, build
    from ._evaluate import evaluate

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

    def print_epoch_metrics(self, e, epochs):
        logs = self.logs
        log_str = "epoch " + str(e) + "/" + str(epochs) + " - loss: "
        log_str += "{:.4f}".format(logs["loss"][-1])
        for metric in self.metrics:
            log_str += " - " + metric.name + ": "
            log_str += "{:.4f}".format(logs[metric.name][-1])
        if self.val_set:
            log_str += " - val_loss: {:.4f}".format(logs["val_loss"][-1])
            for metric in self.metrics:
                log_str += " - val_" + metric.name + ": "
                log_str += "{:.4f}".format(logs["val_" + metric.name][-1])
        print(log_str)

    def reset_gradients(self):
        for layer in self.layers:
            layer.reset_gradients()

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

    def fit(self, features, targets, epochs=10, batch_size=32,
            validation_data=None, callbacks=None):

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
        for e in range(epochs):
            for metric in self.metrics:
                metric.reset_state()
            loss = 0
            bfeatures, btargets, steps = self.get_mini_batchs(
                features,
                targets,
                batch_size
            )
            for s in range(steps):
                loss += self.fit_step([bf[s] for bf in bfeatures], [bt[s] for bt in btargets])
            loss /= steps
            self.update_metrics(loss)
            if validation_data is not None:
                self.evaluate(validation_data)
            self.print_epoch_metrics(e, epochs)
            if callbacks is not None:
                for callback in callbacks:
                    if callback.on_epoch_end(self.logs) is False:
                        return self.logs
        return self.logs

    def summary(self):
        rowsize = 65 if self.linear else 94
        print('_' * rowsize)
        print('{}{}{}{}\n{}'.format(
            'Layer (type)'.ljust(29),
            'Output Shape'.ljust(26),
            'Param #'.ljust(13),
            'Connected To'.ljust(29) if not self.linear else '',
            '=' * rowsize
        ))
        total_param = 0
        for layer in self.flatten_graph:
            param = 0
            param += np.sum(
                [np.prod(w.shape) for w in layer.get_trainable_weights()],
                dtype=int
            )
            if layer is self.flatten_graph[-1]:
                end_str = '=' * rowsize
            else:
                end_str = '_' * rowsize
            if isinstance(layer.output_shape, int):
                output_shape = (layer.output_shape,)
            else:
                output_shape = layer.output_shape
            if layer.input_connectors is not None:
                connected_layer = layer.input_connectors.layer.name
            else:
                connected_layer = ''
            print('{}{}{}{}\n{}'.format(
                f'{layer.name} ({layer.__class__.__name__})'.format().ljust(29),
                f'{(None, *output_shape)}'.ljust(26),
                str(param).ljust(13),
                connected_layer if not self.linear else '',
                end_str
            ))
            total_param += param
        print('Total params:', '{:,}'.format(total_param))
        print('_' * rowsize)

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
                'set weight error: New weights does not contain as many matrix as expected'
                f'expected: {len(self.weights)} matrix'
                f'actual: {len(new_weights)} matrix'
            )
        for weight, new_weight in zip(self.weights, new_weights):
            weight.fill(0)
            weight += new_weight
            # weight[...] = new_weight[...]