import numpy as np

from Protodeep.layers.Layer import Layer


class Input(Layer):

    total_instance = 0

    def __init__(self, shape=None, name='input'):
        super().__init__(trainable=False, name=name)
        self.shape = shape
        self.output_shape = (
            self.shape
        )
        self.output_connectors = self.new_output_connector(
            self.output_shape
        )

    def __call__(self, connectors=None):
        self.input_connectors = connectors
        return self.output_connectors

    def forward_pass(self, inputs):
        self.a_val = np.array(inputs)
        return self.a_val
