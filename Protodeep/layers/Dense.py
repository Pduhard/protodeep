import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(func):
        return func

from Protodeep.utils.parse import parse_activation, parse_initializer
from Protodeep.utils.debug import class_timer
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

# @njit
def dense_preactiv(inputs, weights, biases):
    return np.dot(inputs, weights) + biases


@class_timer
class Dense(Layer):

    total_instance = 0
    weights = None
    w_grad = None

    biases = None
    b_grad = None

    a_val = None
    z_val = None
    i_val = None
    dloss = None

    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None):
        name = 'dense'
        if self.__class__.total_instance > 0:
            name += '_' + str(self.__class__.total_instance)
        super().__init__(trainable=True, name=name)
        self.__class__.total_instance += 1
        self.units = units
        self.activation = parse_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = parse_initializer(kernel_initializer)
        self.bias_initializer = parse_initializer(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.output_shape = (
            # None,
            self.units
        )

    def __call__(self, connectors):
        if isinstance(connectors.shape, tuple):
            connectors.shape = connectors.shape[-1]
        weight_shape = (connectors.shape, self.units)
        self.weights = self.kernel_initializer(weight_shape)
        self.w_grad = np.zeros(weight_shape)
        self.biases = self.bias_initializer(self.units)
        self.b_grad = np.zeros(self.units)

        self.input_connectors = connectors
        self.output_connectors = self.new_output_connector(
            self.output_shape
        )
        return self.output_connectors

    def compile(self, input_shape):
        """ input must be a 1d array """
        if isinstance(input_shape, tuple):
            input_shape = input_shape[-1]
        weight_shape = (input_shape, self.units)
        self.weights = self.kernel_initializer(weight_shape)
        self.w_grad = np.zeros(weight_shape)
        self.biases = self.bias_initializer(self.units)
        self.b_grad = np.zeros(self.units)

        # print(self.weights.shape)
        # print(self.biases.shape)
        # print("------------")
        # return self.units

    def reset_gradients(self):
        self.w_grad.fill(0)
        self.b_grad.fill(0)

    def forward_pass(self, inputs):
        self.i_val = inputs
        self.z_val = dense_preactiv(inputs, self.weights, self.biases)
        self.a_val = self.activation(self.z_val)
        return self.a_val

    def backward_pass(self, inputs):
        a_dp = self.activation.derivative(self.z_val)
        z_dp = inputs * a_dp

        self.w_grad += np.outer(z_dp, self.i_val).T
        self.b_grad += z_dp
        self.dloss = np.matmul(self.weights, z_dp)
        return self.dloss

    def get_weights(self):
        return [self.weights, self.biases]

    def set_weights(self, weights):
        if len(weights) != 2:
            print('chelouuuu set weights dans dense')
        self.weights = weights[0]
        self.biases = weights[1]
