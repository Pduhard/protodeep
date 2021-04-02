import numpy as np
try:
    from numba import njit, prange
except ImportError:
    def njit(func):
        return func
# import matplotlib.pyplot as plt


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
                    # print(self.z_index[oh, ow, f])
                # quit()
                ow += 1
            oh += 1


@njit(parallel=True, fastmath=True)
def maxpool_derivative(z_index, inputs, dx, N, H, W, F):
    for n in prange(N):
        for h in range(H):
            for w in range(W):
                for f in range(F):
                    dh, dw, df = z_index[n, h, w, f]
                    # print(index)
                    # print(inputs[h, w, f])
                    dx[n, dh, dw, df] += inputs[n, h, w, f]

@class_timer
class MaxPool2D(Layer):

    total_instance = 0

    def __init__(self, pool_size=(2, 2), strides=None,
                 padding='valid', data_format=None):
        name = 'maxpool2d'
        if self.__class__.total_instance > 0:
            name += '_' + str(self.__class__.total_instance)
        super().__init__(trainable=False, name=name)

        self.b_mask = None
        self.z_val = None
        self.z_index = None
        self.a_val = None
        self.dloss = None

        self.__class__.total_instance += 1
        self.pool_size = pool_size
        self.strides = strides or pool_size
        self.padding = padding
        self.data_format = data_format

    def __call__(self, connectors):
        h, w, z = connectors.shape
        ph, pw = self.pool_size
        sh, sw = self.strides

        self.output_shape = (
            # None,
            int((h - ph) / sh + 1),  # need to check what append when not round number
            int((w - pw) / sw + 1),
            z,
        )
        self.input_connectors = connectors
        self.output_connectors = self.new_output_connector(
            self.output_shape
        )
        return self.output_connectors

    def compile(self, input_shape):
        print("compile MaxPool2D")
        # print(input_shape)

        h, w, z = input_shape
        ph, pw = self.pool_size
        sh, sw = self.strides

        self.output_shape = (
            # None,
            int((h - ph) / sh + 1),  # need to check what append when not round number
            int((w - pw) / sw + 1),
            z,
        )       
        # print(self.output_shape)
        # quit()
        # return self.units

    def forward_pass(self, inputs):
        # print(inputs.shape)
        # print(len(inputs.shape))
        # quit()
        self.i_val = inputs
        N, H, W, F = inputs.shape
        PH, PW = self.pool_size
        SH, SW = self.strides
        output_shape = (
            N,
            int((H - PH) / SH + 1),  # need to check what append when not round number
            int((W - PW) / SW + 1),
            F,
        )   
        self.z_val = np.zeros(shape=output_shape)
        self.z_index = np.zeros(shape=(*output_shape, len(inputs.shape) - 1), dtype=int)
        # self.z_index[0, 0, 0] = (0,1,2)
        # print(self.z_index[0, 0])
        # quit()
        # print(self.z_val.shape)
        # quit()
        # oh = 0
        # plt.imshow(inputs.T[0].T, cmap='gray')
        # plt.show()
        maxpool(self.z_val, self.z_index, inputs, N, H, PH, W, PW, SH, SW, F)

        # for h in range(0, H - PH, SH):
        #     ow = 0
        #     for w in range(0, W - PW, SW):
        #         for f in range(F):
        #             max_value = -np.inf
        #             max_index = None
        #             for ph in range(PH):
        #                 for pw in range(PW):
        #                     value = inputs[h + ph, w + pw, f]
        #                     if value > max_value:
        #                         max_value = value
        #                         max_index = (h + ph, w + ph, f)
        #             self.z_val[oh, ow, f] = max_value
        #             self.z_index[oh, ow, f] = max_index
        #             # print(self.z_index[oh, ow, f])
        #         # quit()
        #         ow += 1
        #     oh += 1

        # print(self.z_val.shape)
        self.a_val = self.z_val
        return self.a_val

    def backward_pass(self, inputs):

        # plt.imshow(inputs.T[0].T, cmap='gray')
        # plt.show()
        # print(self.z_index.shape)
        # print(self.i_val.shape)

        # print(inputs.shape)
        N, H, W, F = inputs.shape
        dx = np.zeros(self.i_val.shape)
        # print(dx[0, 0, 11])
        maxpool_derivative(self.z_index, inputs, dx, N, H, W, F)
        # for h in range(H):
        #     for w in range(W):
        #         for f in range(F):
        #             dh, dw, df = self.z_index[h, w, f]
        #             # print(index)
        #             # print(inputs[h, w, f])
        #             dx[dh, dw, df] += inputs[h, w, f]
        # plt.imshow(dx.T[0].T, cmap='gray')
        # plt.show()
        # print(dx.shape)
        # quit()
        self.dloss = dx
        return [], self.dloss
        # a_dp = self.activation.derivative(self.z_val)
        # z_dp = inputs * a_dp

        # self.w_grad += np.outer(z_dp, self.i_val).transpose()
        # self.b_grad += z_dp
        # return np.matmul(self.weights, z_dp)
