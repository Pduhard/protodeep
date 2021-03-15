from Protodeep.activations.Activation import Activation
import numpy as np
try:
    from numba import njit, jit
except ImportError:
    def njit(func):
        return func

epsilon = 1e-8


# @jit
# @vectorize


# @jit
# def vsoftmax(x, xsum):
#     return np.exp(x) / (xsum + epsilon)

@njit
def softmax(inputs):  # need to verify when dim != 2
    # print(np.max(np.abs(inputs)))
    # quit()
    # while (np.max(np.abs(inputs)) > 10):
    #     inputs /= 2
    # if len(inputs.shape )
    # xsum = np.array([np.sum(np.exp(i)) for i in inputs])
    # for i in range()
    # print(xsum)
    # xsum = np.sum(np.exp(inputs), axis=1)[:, np.newaxis]
    # print((np.exp(inputs) / (xsum + epsilon)).shape)
    # print(xsum.shape)
    # quit()
    # exp(x) / exp(X)
    sft = np.empty(inputs.shape)
    for i in range(inputs.shape[0]):
        while (np.max(np.abs(inputs[i])) > 10):
            inputs[i] /= 2
        for j in range(inputs.shape[1]):
            sft[i, j] = np.exp(inputs[i, j]) / (np.sum(np.exp(inputs[i])))
    # print(sft.dtype)
    return sft
    # return np.exp(inputs) / xsum[:, np.newaxis] + epsilon
    # return np.array([np.exp(x) / (xsum + epsilon) for x in inputs])
    # return np.array([np.exp(inputs[i]) / (xsum[i] + epsilon) for i in range(inputs.shape[0])])
    # return (np.vectorize(vsoftmax)(inputs, xsum))


@njit
def softmax_derivative(inputs):
    activ = softmax(inputs)
    return activ * (1 - activ)


class Softmax(Activation):

    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, inputs):
        # print(inputs.shape)
        # quit()
        return softmax(inputs)

    def derivative(self, inputs):
        return softmax_derivative(inputs)
