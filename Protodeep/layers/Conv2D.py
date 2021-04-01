import numpy as np
try:
    from numba import njit, prange
except ImportError:
    def njit(func):
        return func
# import matplotlib.pyplot as plt

from Protodeep.layers.Layer import Layer
from Protodeep.utils.parse import parse_activation, parse_initializer, parse_regularizer
from Protodeep.utils.debug import class_timer
from Protodeep.utils.error import _shape_error

@class_timer
@njit(parallel=True, fastmath=True)
def conv(z_val, N, H, W, F, C, KH, KW, SH, SW, weights, biases, inputs):
    for n in prange(N):
        for h in range(0, H - KH, SH):
            for w in range(0, W - KW, SW):
                for f in range(F):
                    z_val[n, h // SH, w // SW, f] = 0
                    for kh in range(KH):
                        for kw in range(KW):
                            for c in range(C):
                                z_val[n, h // SH, w // SW, f] += inputs[n, h + kh, w + kw, c] * weights[kh, kw, c, f]
                    z_val[n, h // SH, w // SW, f] += biases[f]

# @class_timer
# # @njit(parallel=True, fastmath=True)
# def old_conv_derivative(w_grad, b_grad, N, H, W, F, C, KH, KW, a_dp, i_val):
#     for n in prange(N):
#         for h in range(0, H):
#             for w in range(0, W):
#                 for f in range(F):
#                     for kh in range(KH):
#                         for kw in range(KW):
#                             for c in range(C):
#                                 b_grad[c] += a_dp[n, kh, kw, c]
#                                 w_grad[h, w, c, f] += i_val[n, kh + h, kw + w, c] * a_dp[n, kh, kw, c]


@class_timer
@njit(parallel=True, fastmath=True)
def conv_derivative(w_grad, b_grad, N, H, W, F, C, KH, KW, a_dp, i_val):
    for n in prange(N):
        for h in range(0, H):
            for w in range(0, W):
                for c in range(C):
                    for kh in range(KH):
                        for kw in range(KW):
                            for f in range(F):
                                b_grad[f] += a_dp[n, kh, kw, f]
                                w_grad[h, w, c, f] += i_val[n, kh + h, kw + w, c] * a_dp[n, kh, kw, f]
    w_grad /= N
    b_grad /= N

@class_timer
@njit(parallel=True, fastmath=True)
def conv_xgrad(x_grad, N, H, W, F, C, KH, KW, pad_a_dp, weights):
    for n in prange(N):
        for h in range(0, H - KH):
            for w in range(0, W - KW):
                for f in range(F):
                    for kh in range(KH):
                        # th = KH - kh - 1
                        for kw in range(KW):
                            # tw = KW - kw - 1
                            for c in range(C):
                                # print((tw, th, c, f))

                                x_grad[n, h, w, c] += pad_a_dp[n, h + kh, w + kw, f] * weights[kh, kw, c, f]  # 180 rotated
                                # old version :
                                # x_grad[n, h, w, c] += pad_a_dp[n, h + kh, w + kw, c] * weights[tw, th, c, f]  # 180 rotated

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
    # print(arr.shape)
    # print(dilated.shape)
    # quit()
    return dilated


@class_timer
@njit(parallel=True, fastmath=True)
def rotate_180(arr, H, W):
    rotated = np.empty(arr.shape)
    for i in prange(H):
        for j in range(W):
            rotated[i, W-1-j] = arr[H-1-i, j]
    return rotated


@class_timer
class Conv2D(Layer):

    total_instance = 0
    weights = None
    w_grad = None

    biases = None
    b_grad = None

    a_val = None
    z_val = None
    i_val = None
    dloss = None

    def __init__(self, filters, kernel_size, strides=(1, 1),
                 activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None):
        name = 'conv2d'
        if self.__class__.total_instance > 0:
            name += '_' + str(self.__class__.total_instance)
        super().__init__(trainable=True, name=name)
        self.__class__.total_instance += 1
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

    # def compile(self, input_shape):
    #     print('nopeee')
    #     quit()
    #     """ input should be a 3d array (H, W, C)
    #             ( should add batch size here ? )
    #             H = height
    #             W = width
    #             C = channel
    #         weight shape (H, W, C, F)
    #             F = filter
    #     """
        
    #     weight_shape = (*self.kernel_size, input_shape[-1], self.filters)

    #     self.weights = self.kernel_initializer(weight_shape)
    #     self.w_grad = np.zeros(weight_shape)
     
    #     if self.use_bias:
    #         self.biases = self.bias_initializer(self.filters)
    #         self.b_grad = np.zeros(self.filters)


    #     h, w, c = input_shape
    #     kh, kw = self.kernel_size
    #     sh, sw = self.strides

    #     self.output_shape = (
    #         int((h - kh) / sh + 1),
    #         int((w - kw) / sw + 1),
    #         self.filters,
    #     )


    # def reset_gradients(self):
    #     self.w_grad.fill(0)
    #     self.b_grad.fill(0)
        # self.dloss.fill(0)

    def forward_pass(self, inputs):
        """
        input must be a 4d array :
            ( batch_size, height, width, channel )
        """

        # if inputs.shape[1:] != self.input_connectors.shape:
        #     _shape_error(self.input_connectors.shape, inputs.shape[1:])
        
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
        conv(self.z_val, N, H, W, F, C, KH, KW, SH,
             SW, self.weights, self.biases, inputs)
        self.a_val = self.activation(self.z_val)
        return self.a_val

    def backward_pass(self, inputs):

        if self.activity_regularizer:
            inputs = inputs + self.activity_regularizer.derivative(inputs)
        a_dp = self.activation.derivative(self.z_val) * inputs
        if not any(self.strides) > 1:
            a_dp = dilate(a_dp, self.strides[0], self.strides[1])

        self.w_grad.fill(0)
        self.b_grad.fill(0)
        self.dloss = np.zeros(shape=self.i_val.shape)

        H, W, C, F = self.weights.shape
        SH, SW = self.strides
        N, KH, KW, _ = a_dp.shape
        
        # convolution of i_val by a_dp
        conv_derivative(self.w_grad, self.b_grad, N, H, W,
                        F, C, KH, KW, a_dp, self.i_val)

        h_pad = (self.dloss.shape[1] - a_dp.shape[1])
        w_pad = (self.dloss.shape[2] - a_dp.shape[2])
        padding = (
                (0, 0),
                (h_pad, h_pad),
                (w_pad, w_pad),
                (0, 0)
        )
        pad_a_dp = np.pad(a_dp, padding, 'constant')

        N, H, W, _ = pad_a_dp.shape
        KH, KW = self.kernel_size
        rotated = (self.weights[:, ::-1, ...])[::-1, ...]
        conv_xgrad(self.dloss, N, H, W, F, C, KH, KW, pad_a_dp, rotated)

        if self.kernel_regularizer:
            self.w_grad += self.kernel_regularizer.derivative(self.weights)
        if self.bias_regularizer:
            self.b_grad += self.bias_regularizer.derivative(self.biases)

        return [self.w_grad, self.b_grad], self.dloss

    def get_trainable_weights(self):
        return [self.weights, self.biases]

    def get_gradients(self):
        return [self.w_grad, self.b_grad]
