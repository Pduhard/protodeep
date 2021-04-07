import numpy as np
# from numba import njit

from Protodeep.utils.parse import parse_activation
from Protodeep.utils.parse import parse_initializer, parse_regularizer
from Protodeep.utils.debug import class_timer
from Protodeep.layers.Layer import Layer

# @njit
# def dense_preactiv(inputs, weights, biases):
#     return np.dot(inputs, weights) + biases


# !!! ceci est de la grosse merde jpp too slow
# @njit
# def backward(w_grad, b_grad, inputs, a_dp, i_val, weights, batch_size):
#     w_grad.fill(0)
#     b_grad.fill(0)
#     z_dp = (inputs * a_dp).T
#     w_grad += (z_dp @ i_val).T / batch_size
#     for i in range(batch_size):
#         b_grad[i] += z_dp[i]
#     return (weights @ z_dp).T


@class_timer
class Dense(Layer):

    total_instance = 0

    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None,
                 name='dense'):
        super().__init__(trainable=True, name=name)
        self.weights = None
        self.w_grad = None

        self.biases = None
        self.b_grad = None

        self.a_val = None
        self.z_val = None
        self.i_val = None
        self.dloss = None

        self.units = units
        self.activation = parse_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = parse_initializer(kernel_initializer)
        self.bias_initializer = parse_initializer(bias_initializer)
        self.kernel_regularizer = parse_regularizer(kernel_regularizer)
        self.bias_regularizer = parse_regularizer(bias_regularizer)
        self.activity_regularizer = parse_regularizer(activity_regularizer)
        self.output_shape = (
            self.units
        )

    def __call__(self, connectors):

        if isinstance(connectors.shape, tuple):
            connectors.shape = connectors.shape[-1]
        weight_shape = (connectors.shape, self.units)
        self.weights = self.kernel_initializer(weight_shape)
        self.w_grad = np.zeros(weight_shape)

        if self.use_bias:
            self.biases = self.bias_initializer(self.units)
            self.b_grad = np.zeros(self.units)

        self.input_connectors = connectors
        self.output_connectors = self.new_output_connector(
            self.output_shape
        )
        return self.output_connectors

    def reset_gradients(self):
        self.w_grad.fill(0)
        if self.use_bias:
            self.b_grad.fill(0)

    def regularize(self):
        if self.kernel_regularizer:
            self.w_grad += self.kernel_regularizer.derivative(self.weights)
        if self.use_bias and self.bias_regularizer:
            self.b_grad += self.bias_regularizer.derivative(self.biases)

    def forward_pass(self, inputs):
        self.i_val = inputs
        if self.use_bias:
            self.z_val = np.dot(inputs, self.weights) + self.biases
        else:
            self.z_val = np.dot(inputs, self.weights)
        self.a_val = self.activation(self.z_val)
        return self.a_val

    def backward_pass(self, inputs):
        """
            inputs: derivative of loss with respect to output of this layer

            outputs:
                list of gradients (same order as get_trainable_weights),
                and derivative of loss with respect to input of this layer
        """
        # self.dloss = backward(self.w_grad, self.b_grad, inputs,
        # self.activation.derivative(self.z_val), self.i_val,
        # self.weights, inputs.shape[0])

        if self.activity_regularizer:
            inputs = inputs + self.activity_regularizer.derivative(inputs)

        self.reset_gradients()
        a_dp = self.activation.derivative(self.z_val)
        z_dp = (inputs * a_dp).T

        self.w_grad += (z_dp @ self.i_val).T / inputs.shape[0]
        if self.use_bias:
            self.b_grad += np.mean(z_dp, axis=-1)

        self.dloss = (self.weights @ z_dp).T

        self.regularize()
        return self.get_gradients(), self.dloss

    def get_trainable_weights(self):
        return [self.weights, self.biases] if self.use_bias else [self.weights]

    def get_gradients(self):
        return [self.w_grad, self.b_grad] if self.use_bias else [self.w_grad]

    def set_weights(self, weights):
        if len(weights) != len(self.get_trainable_weights()):
            print('invalid weights list dense')
        self.weights = weights[0]
        if self.use_bias:
            self.biases = weights[1]
