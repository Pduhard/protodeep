import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(func):
        return func
# import matplotlib.pyplot as plt

from Protodeep.layers.Layer import Layer
from Protodeep.utils.parse import parse_activation, parse_initializer
from Protodeep.utils.debug import class_timer


@class_timer
@njit
def conv(z_val, H, W, F, C, KH, KW, SH, SW, weights, biases, inputs):
    for h in range(0, H - KH, SH):
        for w in range(0, W - KW, SW):
            for f in range(F):
                z_val[h // SH, w // SW, f] = 0
                for kh in range(KH):
                    for kw in range(KW):
                        for c in range(C):
                            z_val[h // SH, w // SW, f] += inputs[h + kh, w + kw, c] * weights[kh, kw, c, f]
                z_val[h // SH, w // SW, f] += biases[f]

@class_timer
@njit
def conv_derivative(w_grad, b_grad, H, W, F, C, KH, KW, a_dp, i_val):
    for h in range(0, H):
        for w in range(0, W):
            for f in range(F):
                for kh in range(KH):
                    for kw in range(KW):
                        for c in range(C):
                            b_grad[c] += a_dp[kh, kw, c]
                            # w_grad[h, kh, ]
                            w_grad[h, w, c, f] += i_val[kh + h, kw + w, c] * a_dp[kh, kw, c]


@class_timer
@njit
def conv_xgrad(x_grad, H, W, F, C, KH, KW, pad_a_dp, weights):
    for h in range(0, H - KH):
        for w in range(0, W - KW):
            for f in range(F):
                for kh in range(KH):
                    th = KH - kh - 1
                    for kw in range(KW):
                        tw = KW - kw - 1
                        for c in range(C):
                            x_grad[h, w, c] += pad_a_dp[h + kh, w + kw, c] * weights[tw, th, c, f]  # 180 rotated

@class_timer
@njit
def dilate(arr, SH, SW):
    H, W, C = arr.shape
    dilated = np.zeros(((H - 1) * (SH - 1) + H, (W - 1) * (SW - 1) + W, C))
    for h in range(H):
        for w in range(W):
            for c in range(C):
                dilated[h * SH, w * SW, c] = arr[h, w, c]
    # print(arr.shape)
    # print(dilated.shape)
    # quit()
    return dilated

from Protodeep.utils.debug import class_timer

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

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid',
                 data_format=None, dilation_rate=(1, 1), groups=1,
                 activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None):
        name = 'conv2d'
        if self.__class__.total_instance > 0:
            name += '_' + str(self.__class__.total_instance)
        super().__init__(trainable=True, name=name)
        self.__class__.total_instance += 1
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.groups = groups
        self.activation = parse_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = parse_initializer(kernel_initializer)
        self.bias_initializer = parse_initializer(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        # self.output_shape =

    def __call__(self, connectors):
        h, w, c = connectors.shape
        kh, kw = self.kernel_size
        sh, sw = self.strides

        weight_shape = (*self.kernel_size, connectors.shape[-1], self.filters)
        self.weights = self.kernel_initializer(weight_shape)
        # self.weights = np.full(weight_shape, 1)
        self.w_grad = np.zeros(weight_shape)

        self.biases = self.bias_initializer(self.filters)
        self.b_grad = np.zeros(self.filters)

        self.output_shape = (
            # None,
            int((h - kh) / sh + 1),  # need to check what append when not round number
            int((w - kw) / sw + 1),
            self.filters,
        )
        self.input_connectors = connectors
        self.output_connectors = self.new_output_connector(
            self.output_shape
        )
        return self.output_connectors

    def compile(self, input_shape):
        """ input should be a 3d array (H, W, C)
                ( should add batch size here ? )
                H = height
                W = width
                C = channel
            weight shape (H, W, C, F)
                F = filter
        """
        weight_shape = (*self.kernel_size, input_shape[-1], self.filters)
        self.weights = self.kernel_initializer(weight_shape)
        # self.weights = np.full(weight_shape, 1)
        self.w_grad = np.zeros(weight_shape)

        self.biases = self.bias_initializer(self.filters)
        self.b_grad = np.zeros(self.filters)

        h, w, c = input_shape
        kh, kw = self.kernel_size
        sh, sw = self.strides

        self.output_shape = (
            # None,
            int((h - kh) / sh + 1),  # need to check what append when not round number
            int((w - kw) / sw + 1),
            self.filters,
        )

    def reset_gradients(self):
        self.w_grad.fill(0)
        self.b_grad.fill(0)

    def forward_pass(self, inputs):
        """ input should be a 3d array (H, W, C) """
        
        # plt.imshow(inputs, cmap=plt.get_cmap('gray'))
        # plt.show()
        # print(inputs)
        if len(inputs.shape) < 3:
            inputs = inputs[:, :, np.newaxis]
        
        # plt.imshow(inputs.reshape(28, 28), cmap=plt.get_cmap('gray'))
        # plt.show()
        # print(inputs)
        # print(inputs.shape)
        # print(len(inputs.shape))
        # quit()
        H, W, C = inputs.shape
        SH, SW = self.strides
        KH, KW = self.kernel_size
        F = self.filters
        # !!!!! padding not implemented
        self.i_val = inputs
        # print (self.output_shape)
        self.z_val = np.zeros(shape=self.output_shape)
        # print(self.z_val.shape)
        # quit()
        conv(self.z_val, H, W, F, C, KH, KW, SH, SW, self.weights, self.biases, inputs)
        #     ow += 1
        # oh += 1
        
        # print(np.sum(self.z_val.T[5]))
        # plt.imshow(self.z_val.T[25].T, cmap=plt.get_cmap('gray'))
        # plt.show()
        # self.z_val = np.matmul(inputs, self.weights) + self.biases
        self.a_val = self.activation(self.z_val)
        return self.a_val

    def backward_pass(self, inputs):
        # print('dl shape =', inputs.shape)
        # print('w shape =', self.weights.shape)
        # print('b shape =', self.biases.shape)
        a_dp = self.activation.derivative(self.z_val) * inputs
        # test = np.arange(9).reshape(3, 3, 1)
        # print(test.reshape(3, 3))
        # print(dilate(test, *self.strides).reshape(5, 5))
        # quit()
        # print(a_dp.T[0].T)
        if not any(self.strides) > 1:
            a_dp = dilate(a_dp, self.strides[0], self.strides[1])

        # print(a_dp.T[0].T)
        # quit()
        # z_dp = np.zeros(shape=self.weights.shape)
        # w_grad = np.zeros(shape=self.weights.shape)
        b_grad = np.zeros(shape=self.biases.shape)
        x_grad = np.zeros(shape=self.i_val.shape)
        # print('a shape =', a_dp.shape)
        # print('x shape =', x_grad.shape)

        H, W, C, F = self.weights.shape
        SH, SW = self.strides
        KH, KW, _ = a_dp.shape
        
        # convolution of i_val by a_dp
        conv_derivative(self.w_grad, self.b_grad, H, W, F, C, KH, KW, a_dp, self.i_val)
        # print(self.w_grad.shape)
        # print(self.i_val.shape)
        # print(a_dp.shape)
        # print(H, W, F, C, KH, KW)
        # quit()
                                # pond_sum += inputs[h + kh, w + kw, c] * self.weights[kh, kw, c, f]
                    # self.z_val[oh, ow, f] += pond_sum + self.biases[f]
            #     ow += 1
            # oh += 1

        # z_dp = inputs * a_dp

        # print(a_dp.shape)
        h_pad = (x_grad.shape[0] - a_dp.shape[0])
        w_pad = (x_grad.shape[1] - a_dp.shape[1])
        padding = (
                (h_pad, h_pad),
                (w_pad, w_pad),
                (0, 0)
        )
        # print(padding)
        pad_a_dp = np.pad(a_dp, padding, 'constant')
        # print('-----------')
        # print(x_grad.shape)
        # print(a_dp.shape)
        # print(pad_a_dp.shape)
        # print(self.weights.shape)

        H, W, _ = pad_a_dp.shape
        # sh, sw = self.strides
        KH, KW = self.kernel_size
        # _, _, C, F = self.weights.shape
        # print(x_grad.shape, H, W, F, C, KH, KW*, pad_a_dp.shape, self.weights.shape)
        conv_xgrad(x_grad, H, W, F, C, KH, KW, pad_a_dp, self.weights)

        # self.weights
        # print(x_grad.shape)
        # quit()
        self.dloss = x_grad
        return self.dloss
        # return np.matmul(self.weights, z_dp)