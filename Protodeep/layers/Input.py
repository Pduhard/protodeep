import numpy as np


from Protodeep.utils.parse import parse_activation, parse_initializer
from Protodeep.utils.debug import timer, class_timer
from Protodeep.layers.Layer import Layer
# def parse_initializer(initializer):
#     if isinstance(initializer, str) is False:
#         return initializer
#     initializer = initializer.lower()
#     if initializer == "henormal":
#         return HeNormal()
#     elif initializer == "glorotnormal":
#         return GlorotNormal()
#     elif initializer == "randomnormal":
#         return RandomNormal()
#     else:
#         return HeNormal()


# def parse_activation(activation):
#     if isinstance(activation, str) is False:
#         return activation
#     activation = activation.lower()
#     if activation == "softmax":
#         return Softmax()
#     elif activation == "relu":
#         return Relu()
#     else:
#         return Relu()  # !! linear

# @class_timer
class Input(Layer):

    total_instance = 0
    # weights = None
    # w_grad = None

    # biases = None
    # b_grad = None

    # a_val = None
    # z_val = None
    # i_val = None
    # dloss = None

    def __init__(self, shape=None):
        name = 'input'
        if self.__class__.total_instance > 0:
            name += '_' + str(self.__class__.total_instance)
        super().__init__(trainable=False, name=name)
        self.__class__.total_instance += 1
        self.shape = shape
        self.output_shape = (
            self.shape
        )
        self.output_connectors = self.new_output_connector(
            self.output_shape
        )

    def __call__(self, connectors=None):
        self.input_connectors = connectors
        # self.output_connectors = self.new_output_connector(
        #     self.output_shape
        # )
        return self.output_connectors
    # def compile(self, input_shape):
    #     pass
        # """ input must be a 1d array """
        # if isinstance(input_shape, tuple):
        #     input_shape = input_shape[-1]
        # weight_shape = (input_shape, self.units)
        # self.weights = self.kernel_initializer(weight_shape)
        # self.w_grad = np.zeros(weight_shape)
        # self.biases = self.bias_initializer(self.units)
        # self.b_grad = np.zeros(self.units)

        # print(self.weights.shape)
        # print(self.biases.shape)
        # print("------------")
        # return self.units

    # def reset_gradients(self):
    #     self.w_grad.fill(0)
    #     self.b_grad.fill(0)

    # # @timer
    def forward_pass(self, inputs):
        self.a_val = np.array(inputs)
        return self.a_val

    # def backward_pass(self, inputs):
    #     a_dp = self.activation.derivative(self.z_val)
    #     z_dp = inputs * a_dp

    #     self.w_grad += np.outer(z_dp, self.i_val).transpose()
    #     self.b_grad += z_dp
    #     return np.matmul(self.weights, z_dp)
