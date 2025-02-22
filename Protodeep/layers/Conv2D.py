import numpy as np
from numba import njit, prange

from Protodeep.layers.Layer import Layer
from Protodeep.utils.parse import parse_activation
from Protodeep.utils.parse import parse_initializer, parse_regularizer
from Protodeep.utils.debug import class_timer
from Protodeep.utils.error import _ndim_error


@class_timer
@njit(parallel=True, fastmath=True)
def conv(z_val, N, H, W, F, C, KH, KW, SH, SW, weights, biases, inputs):
    for n in prange(N):
        for h in range(0, H - KH, SH):
            for w in range(0, W - KW, SW):
                for f in range(F):
                    z_val[n, h // SH, w // SW, f] += np.sum(inputs[n, h:h+KH, w:w+KW, :] * weights[:, :, :, f]) + biases[f]


@class_timer
@njit(parallel=True, fastmath=True)
def conv_no_bias(z_val, N, H, W, F, C, KH, KW, SH, SW, weights, inputs):
    for n in prange(N):
        for h in range(0, H - KH, SH):
            for w in range(0, W - KW, SW):
                for f in range(F):
                    z_val[n, h // SH, w // SW, f] += np.sum(inputs[n, h:h+KH, w:w+KW, :] * weights[:, :, :, f])

# slower...

# @class_timer
# @njit(parallel=True, fastmath=True)
# def conv(z_val, N, H, W, F, C, KH, KW, SH, SW, weights, biases, inputs):
#     for n in prange(N):
#         for h in range(0, H - KH, SH):
#             for w in range(0, W - KW, SW):
#                 for f in range(F):
#                     for kh in range(KH):
#                         for kw in range(KW):
#                             for c in range(C):
#                                 z_val[n, h // SH, w // SW, f] += inputs[n, h + kh, w + kw, c] * weights[kh, kw, c, f]
#                     z_val[n, h // SH, w // SW, f] += biases[f]


# @class_timer
# @njit(parallel=True, fastmath=True)
# def conv_no_bias(z_val, N, H, W, F, C, KH, KW, SH, SW, weights, inputs):
#     for n in prange(N):
#         for h in range(0, H - KH, SH):
#             for w in range(0, W - KW, SW):
#                 for f in range(F):
#                     for kh in range(KH):
#                         for kw in range(KW):
#                             for c in range(C):
#                                 z_val[n, h // SH, w // SW, f] += inputs[n, h + kh, w + kw, c] * weights[kh, kw, c, f]


# slower
# @class_timer
# @njit(parallel=True, fastmath=True)
# def conv_derivative(w_grad, b_grad, N, H, W, F, C, KH, KW, a_dp, i_val):
#     for n in prange(N):
#         for h in range(H):
#             for w in range(W):
#                 for c in range(C):
#                     for kh in range(KH):
#                         for kw in range(KW):
#                             for f in range(F):
#                                 b_grad[f] += a_dp[n, kh, kw, f] / N
#                                 w_grad[h, w, c, f] += i_val[n, kh + h, kw + w, c] * a_dp[n, kh, kw, f] / N

# @class_timer
# @njit(parallel=True, fastmath=True)
# def conv_derivative_no_bias(w_grad, N, H, W, F, C, KH, KW, a_dp, i_val):
#     for n in prange(N):
#         for h in range(H):
#             for w in range(W):
#                 for c in range(C):
#                     for kh in range(KH):
#                         for kw in range(KW):
#                             for f in range(F):
#                                 w_grad[h, w, c, f] += i_val[n, kh + h, kw + w, c] * a_dp[n, kh, kw, f] / N


@class_timer
@njit(parallel=True, fastmath=True)
def conv_derivative(w_grad, b_grad, N, H, W, F, C, KH, KW, a_dp, i_val):
    for n in prange(N):
        for h in range(0, H):
            for w in range(0, W):
                for c in range(C):
                    for f in range(F):
                        b_grad[f] += np.sum(a_dp[n, :, :, f]) / N
                        w_grad[h, w, c, f] += np.sum(i_val[n, h:h+KH, w:w+KW, c] * a_dp[n, :, :, f]) / N


@class_timer
@njit(parallel=True, fastmath=True)
def conv_derivative_no_bias(w_grad, N, H, W, F, C, KH, KW, a_dp, i_val):
    for n in prange(N):
        for h in range(0, H):
            for w in range(0, W):
                for c in range(C):
                    for f in range(F):
                        w_grad[h, w, c, f] += np.sum(i_val[n, h:h+KH, w:w+KW, c] * a_dp[n, :, :, f]) / N


@class_timer
@njit(parallel=True, fastmath=True)
def conv_xgrad(x_grad, N, H, W, F, C, KH, KW, pad_a_dp, weights):
    for n in prange(N):
        for h in range(H - KH):
            for w in range(W - KW):
                for f in range(F):
                    for kh in range(KH):
                        for kw in range(KW):
                            for c in range(C):
                                x_grad[n, h, w, c] += pad_a_dp[n, h + kh, w + kw, f] * weights[kh, kw, c, f]


@class_timer
@njit(parallel=True, fastmath=True)
def dilate(arr, SH, SW):
    N, H, W, C = arr.shape
    dilated = np.zeros((N, (H - 1) * (SH - 1) + H, (W - 1) * (SW - 1) + W, C))
    for n in prange(N):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    dilated[n, h * SH, w * SW, c] = arr[n, h, w, c]
    return dilated


def conv_pad(arr, ishape, fshape):
    h_pad = (ishape[1] - fshape[1])
    w_pad = (ishape[2] - fshape[2])
    padding = (
            (0, 0),
            (h_pad, h_pad),
            (w_pad, w_pad),
            (0, 0)
    )
    return np.pad(arr, padding, 'constant')


@class_timer
class Conv2D(Layer):

    total_instance = 0

    def __init__(self, filters, kernel_size, strides=(1, 1),
                 activation=None, use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None,
                 name='conv2d'):
        super().__init__(trainable=True, name=name)

        self.weights = None
        self.w_grad = None

        self.biases = None
        self.b_grad = None

        self.a_val = None
        self.z_val = None
        self.i_val = None
        self.dloss = None

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = parse_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = parse_initializer(kernel_initializer)
        self.bias_initializer = parse_initializer(bias_initializer)
        self.kernel_regularizer = parse_regularizer(kernel_regularizer)
        self.bias_regularizer = parse_regularizer(bias_regularizer)
        self.activity_regularizer = parse_regularizer(activity_regularizer)

    def __call__(self, connectors):

        if len(connectors.shape) != 3:
            print('Conv2D input ndim must be (Width, Height, Channel)')
            _ndim_error(3, len(connectors.shape))

        h, w, c = connectors.shape
        kh, kw = self.kernel_size
        sh, sw = self.strides

        weight_shape = (kh, kw, c, self.filters)
        self.weights = self.kernel_initializer(weight_shape)
        self.w_grad = np.zeros(weight_shape)

        if self.use_bias:
            self.biases = self.bias_initializer(self.filters)
            self.b_grad = np.zeros(self.filters)

        self.output_shape = (
            int((h - kh) / sh + 1),
            int((w - kw) / sw + 1),
            self.filters,
        )
        self.input_connectors = connectors
        self.output_connectors = self.new_output_connector(
            self.output_shape
        )
        return self.output_connectors

    def reset_gradients(self):
        self.w_grad.fill(0)
        if self.use_bias:
            self.b_grad.fill(0)
        self.dloss = np.zeros(shape=(self.i_val.shape))

    def regularize(self):
        if self.kernel_regularizer:
            self.w_grad += self.kernel_regularizer.derivative(self.weights)
        if self.use_bias and self.bias_regularizer:
            self.b_grad += self.bias_regularizer.derivative(self.biases)

    def forward_pass(self, inputs):
        """
        input must be a 4d array :
            ( batch_size, height, width, channel )
        """

        N, H, W, C = inputs.shape
        SH, SW = self.strides
        KH, KW = self.kernel_size
        F = self.filters
        output_shape = (
            N,
            int((H - KH) / SH + 1),
            int((W - KW) / SW + 1),
            F
        )
        self.i_val = inputs
        self.z_val = np.zeros(shape=output_shape)
        if self.use_bias:
            conv(self.z_val, N, H, W, F, C, KH, KW, SH,
                 SW, self.weights, self.biases, inputs)
        else:
            conv_no_bias(self.z_val, N, H, W, F, C, KH,
                         KW, SH, SW, self.weights, inputs)
        self.a_val = self.activation(self.z_val)
        return self.a_val

    def backward_pass(self, inputs):

        if self.activity_regularizer:
            inputs = inputs + self.activity_regularizer.derivative(inputs)

        a_dp = self.activation.derivative(self.z_val) * inputs
        if not any(self.strides) > 1:
            a_dp = dilate(a_dp, self.strides[0], self.strides[1])

        self.reset_gradients()

        # convolution of i_val by a_dp
        H, W, C, F = self.weights.shape
        SH, SW = self.strides
        N, KH, KW, _ = a_dp.shape

        if self.use_bias:
            conv_derivative(self.w_grad, self.b_grad, N, H, W, F, C,
                            KH, KW, a_dp, self.i_val)
        else:
            conv_derivative_no_bias(self.w_grad, N, H, W, F, C,
                                    KH, KW, a_dp, self.i_val)

        # convolution of padded a_dp by 180 rotated weights
        pad_a_dp = conv_pad(a_dp, self.dloss.shape, a_dp.shape)
        N, H, W, _ = pad_a_dp.shape
        KH, KW = self.kernel_size
        conv_xgrad(self.dloss, N, H, W, F, C, KH, KW,
                   pad_a_dp, self.weights[::-1, ::-1, ...])

        self.regularize()
        return self.get_gradients(), self.dloss

    def get_trainable_weights(self):
        return [self.weights, self.biases] if self.use_bias else [self.weights]

    def get_gradients(self):
        return [self.w_grad, self.b_grad] if self.use_bias else [self.w_grad]

    def set_weights(self, weights):
        if len(weights) != len(self.get_trainable_weights()):
            print('invalid weights list conv2d')
        self.weights = weights[0]
        if self.use_bias:
            self.biases = weights[1]
