from Protodeep.utils.parse import parse_optimizer, parse_loss, parse_metrics
from Protodeep.layers.Layer import Layer
from Protodeep.layers.Input import Input


def add(self, layer):
    self.layers.append(layer)


def build_graph(self, inputs, outputs):

    #  need to verify if overidding is working well
    #  but actually no layer take more than 1 input
    def get_next_level(graph, current_level, next_order, depth, way):
        next_level = []
        print(current_level)
        for curr in current_level:
            graph[curr][f'used{way}'] = 1
            for next_l in graph[curr][way]:
                next_level.append(next_l)
                if way == 'nexts':
                    graph[next_l]['order'] = next_order
                    graph[next_l]['depth'] = depth
                next_order += 1
        print(next_level)
        return next_level, next_order

    def order_graph(graph, inputs, outputs):
        """ BFS Heat Map With overidding """
        current = [i.layer.name for i in inputs]
        next_order = 1
        depth = 0
        while len(current) > 0:
            depth += 1
            current, next_order = get_next_level(
                graph,
                current,
                next_order,
                depth,
                'nexts'
            )
        #  run otherway to remove unused layers
        current = [o.layer.name for o in outputs]
        depth = 0
        while len(current) > 0:
            depth += 1
            current, next_order = get_next_level(
                graph,
                current,
                next_order,
                depth,
                'prevs'
            )

    if isinstance(inputs, list) is False:
        inputs = [inputs]
    if isinstance(outputs, list) is False:
        outputs = [outputs]

    ldict = Layer.layer_dico
    for name, layer in ldict.items():
        print(f'{name}: locked:{layer.locked}')
        if not layer.locked:
            for nm, l in ldict.items():
                if name not in self.graph.keys():
                    self.graph[name] = {
                        'prevs': [],
                        'nexts': [],
                        'order': 0,
                        'depth': 0,
                        'usednexts': 0,
                        'usedprevs': 0,
                    }
                if l.input_connectors is layer.output_connectors:
                    self.graph[name]['nexts'].append(nm)
                    layer.output_connectors.next_layers.append(l)
                    if len(self.graph[name]['nexts']) > 1:
                        self.linear = False
                if layer.input_connectors is l.output_connectors:
                    self.graph[name]['prevs'].append(nm)

    order_graph(self.graph, inputs, outputs)
    self.flatten_graph = [
        ldict[layer]
        for layer
        in sorted(self.graph, key=lambda l: self.graph[l]['order']) if self.graph[layer]['usednexts'] == self.graph[layer]['usedprevs'] == 1
    ]
    for layer in self.flatten_graph:
        layer.locked = True

    print(self.graph)
    self.inputs = [i.layer for i in inputs]
    # print(self.inputs)
    self.outputs = [o.layer for o in outputs]
    # print(self.outputs)


def build(self):
    for layer in self.flatten_graph[::-1]:
        self.weights.extend(layer.get_trainable_weights())
        # self.gradients.extend(layer.get_gradients())
        # if layer.trainable is True:
        #     self.weights.append(layer.weights)
        #     self.gradients.append(layer.w_grad)
        #     if layer.use_bias is True:
        #         self.weights.append(layer.biases)
        #         self.gradients.append(layer.b_grad)


def compile(self, features_shape, loss="BinaryCrossentropy",
            optimizer="Adam", metrics=None):
    if self.linked is False:
        input_layer = Input((features_shape))
        connectors = input_layer()
        self.flatten_graph = []
        prev_layer = None
        self.flatten_graph.append(input_layer)
        self.inputs = [input_layer]
        self.outputs = [self.layers[-1]]
        for layer in self.layers:
            connectors = layer(connectors)
            if prev_layer is not None:
                prev_layer.output_connectors.next_layers.append(layer)
            self.flatten_graph.append(layer)
            prev_layer = layer
    self.logs = {}
    self.loss = parse_loss(loss)
    self.optimizer = parse_optimizer(optimizer)
    self.metrics = parse_metrics(metrics)
    self.build()
    self.optimizer.compile(self)
