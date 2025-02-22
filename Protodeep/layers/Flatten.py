import numpy as np

from Protodeep.layers.Layer import Layer
from Protodeep.utils.debug import class_timer


@class_timer
class Flatten(Layer):

    total_instance = 0

    def __init__(self, data_format=None, name='flatten'):
        super().__init__(trainable=False, name=name)
        self.data_format = data_format

    def __call__(self, connectors):
        self.output_shape = (
            np.prod(connectors.shape),
        )
        self.input_connectors = connectors
        self.output_connectors = self.new_output_connector(
            self.output_shape
        )
        return self.output_connectors

    def compile(self, input_shape):
        self.output_shape = (
            np.prod(input_shape),
        )

    def reset_gradients(self):
        pass

    def forward_pass(self, inputs):
        self.input_shape = inputs.shape
        batch_size = inputs.shape[0]
        self.a_val = inputs.reshape(batch_size, int(inputs.size / batch_size))
        return self.a_val

    def backward_pass(self, inputs):
        self.dloss = inputs.reshape(self.input_shape)
        return [], self.dloss
