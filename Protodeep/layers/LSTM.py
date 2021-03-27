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
# def dense_preactiv(inputs, weights, biases):
#     return np.dot(inputs, weights) + biases

# @njit
# def backward(w_grad, b_grad, inputs, a_dp, i_val, weights, batch_size):
#     w_grad.fill(0)
#     b_grad.fill(0)
#     z_dp = (inputs * a_dp).T
#     w_grad += np.matmul(z_dp, i_val).T / batch_size
#     b_grad += np.mean(z_dp, axis=-1)
#     return np.matmul(weights, z_dp).T

@class_timer
class LSTM(Layer):

    total_instance = 0
    weights = None
    w_grad = None

    biases = None
    b_grad = None

    a_val = None
    z_val = None
    i_val = None
    dloss = None

    def __init__(self, units, activation='tanh', recurrent_activation='sigmoid',
                 use_bias=True, kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros', unit_forget_bias=True,
                 kernel_regularizer=None, recurrent_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, recurrent_constraint=None,
                 bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=False, return_state=False,
                 go_backwards=False, stateful=False, time_major=False,
                 unroll=False):
        name = 'lstm'
        if self.__class__.total_instance > 0:
            name += '_' + str(self.__class__.total_instance)
        super().__init__(trainable=True, name=name)
        self.__class__.total_instance += 1
        self.units = units
        self.activation = parse_activation(activation)
        self.recurrent_activation = parse_activation(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = parse_initializer(kernel_initializer)
        self.recurrent_initializer = parse_initializer(recurrent_initializer)
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
        # if isinstance(connectors.shape, tuple):
        #     connectors.shape = connectors.shape[-1]
        # print(connectors.shape[1:])

        weight_shape = (connectors.shape[-1] + self.units, self.units)
        # print(weight_shape)
        # quit()
        self.forget_weight = self.recurrent_initializer(weight_shape)
        self.memory_weight = self.recurrent_initializer(weight_shape)
        self.input_weight = self.kernel_initializer(weight_shape)
        self.output_weight = self.recurrent_initializer(weight_shape)
        self.forget_biase = self.bias_initializer(self.units)
        self.memory_biase = self.bias_initializer(self.units)
        self.input_biase = self.bias_initializer(self.units)
        self.output_biase = self.bias_initializer(self.units)
        # self.weights = self.kernel_initializer(weight_shape)
        self.forget_w_grad = np.zeros(weight_shape)
        self.memory_w_grad = np.zeros(weight_shape)
        self.input_w_grad = np.zeros(weight_shape)
        self.output_w_grad = np.zeros(weight_shape)

        self.forget_b_grad = np.zeros(self.units)
        self.memory_b_grad = np.zeros(self.units)
        self.input_b_grad = np.zeros(self.units)
        self.output_b_grad = np.zeros(self.units)
        # self.biases = self.bias_initializer(self.units)
        # self.b_grad = np.zeros(self.units)

        self.input_connectors = connectors
        self.output_connectors = self.new_output_connector(
            self.output_shape
        )
        return self.output_connectors

    def compile(self, input_shape):
        """ input must be a 1d array """
        print('ha')
        quit()
        if isinstance(input_shape, tuple):
            input_shape = input_shape[-1]
        weight_shape = (input_shape, self.units)
        self.weights = self.kernel_initializer(weight_shape)
        self.w_grad = np.empty(weight_shape)
        self.biases = self.bias_initializer(self.units)
        self.b_grad = np.empty(self.units)
        # self.dloss = np.empty()
        

        # print(self.weights.shape)
        # print(self.biases.shape)
        # print("------------")
        # return self.units

    def init_gradients(self, batch_size):
        self.w_grad = np.empty(self.weights.shape)
        self.b_grad = np.empty(self.biases.shape)
        self.dloss = np.empty((batch_size, self.input_connectors.shape))
        print('dlosss', self.dloss.shape)
        # self.w_grad.fill(0)
        # self.b_grad.fill(0)
        # self.dloss.fill(0)
    
    def reset_gradients(self):
        self.w_grad.fill(0)
        self.b_grad.fill(0)
        self.dloss.fill(0)

    def forward_pass(self, inputs):
        print('forward lstm')
        # print(inputs.shape)
        quit()
        # self.i_val = inputs
        # self.z_val = np.dot(inputs, self.weights) + self.biases
        # self.a_val = self.activation(self.z_val)
        # print(self.a_val)
        return self.a_val

    def backward_pass(self, inputs):
        """
            inputs: derivative of loss with respect to output of this layer

            outputs:
                list of gradients (same order as get_trainable_weights),
                and derivative of loss with respect to input of this layer
        """
        # self.w_grad.fill(0)
        # self.b_grad.fill(0)
        # # self.dloss.fill(0)
        # a_dp = self.activation.derivative(self.z_val)
        # z_dp = (inputs * a_dp).T

        # self.w_grad += np.matmul(z_dp, self.i_val).T / inputs.shape[0]
        # self.b_grad += np.mean(z_dp, axis=-1)
        # # for i in range(inputs.shape[0]):
        #     # self.self.w_grad += np.outer(z_dp[i], self.i_val[i]).T
        #     # self.self.b_grad += z_dp[i]
        #     # dloss.append(np.matmul(self.weights, z_dp[i]))
        #     # self.dloss = np.matmul(self.weights, z_dp).T
        # # self.self.w_grad /= inputs.shape[0]
        # # self.self.b_grad /= inputs.shape[0]
        # self.dloss = np.matmul(self.weights, z_dp).T
        # # self.dloss = np.array(dloss)
        print('bckwrd lstm')
        quit()
        return [self.w_grad, self.b_grad], self.dloss
        

    def get_trainable_weights(self):
        return [
            self.forget_weight,
            self.memory_weight,
            self.input_weight,
            self.output_weight,
            self.forget_biase,
            self.memory_biase,
            self.input_biase,
            self.output_biase
        ]


    def get_gradients(self):
        return [
            self.forget_w_grad,
            self.memory_w_grad,
            self.input_w_grad,
            self.output_w_grad,
            self.forget_b_grad,
            self.memory_b_grad,
            self.input_b_grad,
            self.output_b_grad
        ]
    
    def set_weights(self, weights):
        if len(weights) != 2:
            print('chelouuuu set weights dans dense')
        self.weights = weights[0]
        self.biases = weights[1]
