import numpy as np
from numba import njit, prange


from Protodeep.utils.debug import class_timer
from Protodeep.layers.Layer import Layer


@njit(fastmath=True)
def maxpool(z_val, z_index, inputs, N, H, PH, W, PW, SH, SW, F):
    for n in range(N):
        oh = 0
        for h in range(0, H - PH, SH):
            ow = 0
            for w in range(0, W - PW, SW):
                for f in range(F):
                    max_value = -np.inf
                    max_index = None
                    for ph in range(PH):
                        for pw in range(PW):
                            value = inputs[n, h + ph, w + pw, f]
                            if value > max_value:
                                max_value = value
                                max_index = (h + ph, w + ph, f)
                    z_val[n, oh, ow, f] = max_value
                    z_index[n, oh, ow, f] = max_index
                ow += 1
            oh += 1


@njit(parallel=True, fastmath=True)
def maxpool_derivative(z_index, inputs, dx, N, H, W, F):
    for n in prange(N):
        for h in range(H):
            for w in range(W):
                for f in range(F):
                    dh, dw, df = z_index[n, h, w, f]
                    dx[n, dh, dw, df] += inputs[n, h, w, f]


@class_timer
class MaxPool2D(Layer):

    total_instance = 0

    def __init__(self, pool_size=(2, 2), strides=None,
                 padding='valid', data_format=None, name='maxpool2d'):
        super().__init__(trainable=False, name=name)

        self.b_mask = None
        self.z_val = None
        self.z_index = None
        self.a_val = None
        self.dloss = None

        self.pool_size = pool_size
        self.strides = strides or pool_size
        self.padding = padding
        self.data_format = data_format

    def __call__(self, connectors):
        h, w, z = connectors.shape
        ph, pw = self.pool_size
        sh, sw = self.strides

        self.output_shape = (
            int((h - ph) / sh + 1),
            int((w - pw) / sw + 1),
            z,
        )
        self.input_connectors = connectors
        self.output_connectors = self.new_output_connector(
            self.output_shape
        )
        return self.output_connectors

    def compile(self, input_shape):

        h, w, z = input_shape
        ph, pw = self.pool_size
        sh, sw = self.strides

        self.output_shape = (
            int((h - ph) / sh + 1),
            int((w - pw) / sw + 1),
            z,
        )

    def forward_pass(self, inputs):
        self.i_val = inputs
        N, H, W, F = inputs.shape
        PH, PW = self.pool_size
        SH, SW = self.strides
        output_shape = (
            N,
            int((H - PH) / SH + 1),
            int((W - PW) / SW + 1),
            F,
        )
        self.z_val = np.zeros(shape=output_shape)
        self.z_index = np.zeros((*output_shape, inputs.ndim - 1), dtype=int)
        maxpool(self.z_val, self.z_index, inputs, N, H, PH, W, PW, SH, SW, F)
        self.a_val = self.z_val
        return self.a_val

    def backward_pass(self, inputs):

        N, H, W, F = inputs.shape
        dx = np.zeros(self.i_val.shape)
        maxpool_derivative(self.z_index, inputs, dx, N, H, W, F)
        self.dloss = dx
        return [], self.dloss
