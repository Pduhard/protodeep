from Protodeep.layers.connectors.Connector import Connector


class Layer:

    input_connectors = None
    output_connectors = None
    layer_dico = {}

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False):
        self.trainable = trainable
        self.name = name
        self.dtype = dtype
        self.dynamic = dynamic
        self.layer_dico[name] = self

    def __call__(self, connectors):
        """
            take a Connector obj or list/tuple and set input_connectors to connectors
        """
        self.input_connectors = connectors

    def compile(self, input_shape):
        pass

    def reset_gradients(self):
        pass

    def forward_pass(self, inputs):
        return inputs

    def backward_pass(self, inputs):
        return inputs

    @classmethod
    def print_dico(self):
        for k, v in self.layer_dico.items():
            print(k, ':', v)

    def new_output_connector(self, shapes):
        if isinstance(shapes, list):
            output_connector = []
            for shape in shapes:
                output_connector.append(Connector(shape, self))
            return output_connector
        else:
            return Connector(shapes, self)