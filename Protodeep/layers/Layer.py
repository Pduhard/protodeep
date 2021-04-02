from Protodeep.layers.connectors.Connector import Connector


class Layer:

    layer_dico = {}

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False):
        
        self.input_connectors = None
        self.output_connectors = None
        self.trainable = trainable
        self.name = name
        self.dtype = dtype
        self.dynamic = dynamic
        self.layer_dico[name] = self
        self.locked = False  # used by model class to build ùultiple graphs

    def __call__(self, connectors):
        """
            take a Connector obj or list/tuple and set input_connectors to connectors
        """
        self.input_connectors = connectors

    def compile(self, input_shape):
        pass

    def init_gradients(self, batch_size):
        pass

    def reset_gradients(self):
        pass

    def forward_pass(self, inputs):
        return inputs

    def backward_pass(self, inputs):
        return [], None

    def get_trainable_weights(self):
        return []

    def get_gradients(self):
        return []
    # def set_weights(self, weights):
    #     return None

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
