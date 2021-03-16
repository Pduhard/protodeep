import numpy as np

from Protodeep.layers.Layer import Layer
# from utils.parse import parse_activation, parse_initializer
from Protodeep.utils.debug import class_timer

@class_timer
class Flatten(Layer):

    total_instance = 0
    i_val = None
    z_val = None
    a_val = None
    dloss = None

    input_shape = None

    def __init__(self, data_format=None):
        name = 'flatten'
        if self.__class__.total_instance > 0:
            name += '_' + str(self.__class__.total_instance)
        super().__init__(trainable=False, name=name)
        self.__class__.total_instance += 1
        self.data_format = data_format

    def __call__(self, connectors):
        self.output_shape = (
            # None,
            np.prod(connectors.shape),
        )
        self.input_connectors = connectors
        self.output_connectors = self.new_output_connector(
            self.output_shape
        )
        return self.output_connectors

    def compile(self, input_shape):
        self.output_shape = (
            # None,
            np.prod(input_shape),
        )

        # quit()
        # return self.units

    def reset_gradients(self):
        pass

    def forward_pass(self, inputs):
        self.input_shape = inputs.shape

        self.a_val = inputs.reshape(inputs.sh) nened to reshape
        return self.a_val
        # print(inputs.shape)
        # quit()
        # self.i_val = inputs
        # self.z_val = np.matmul(inputs, self.weights) + self.biases
        # self.a_val = self.activation(self.z_val)
        # return self.a_val

    def backward_pass(self, inputs):
        self.dloss = inputs.reshape(self.input_shape)
        return self.dloss
        # a_dp = self.activation.derivative(self.z_val)
        # z_dp = inputs * a_dp

        # self.w_grad += np.outer(z_dp, self.i_val).transpose()
        # self.b_grad += z_dp
        # return np.matmul(self.weights, z_dp)
