from Protodeep.activations.Activation import Activation
import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(func):
        return func
# def vrelu(x):
#     return max(0, x)


# def relu(inputs):
#   t = np.vectorize(vrelu, otypes=[float])(inputs)
#   print(type(t[0]))
#   return t

# def vrelu_dp(x):
#     return 1 if x > 0 else 0

# def relu_dp(inputs):
#     return np.vectorize(vrelu_dp, otypes=[float])(inputs)


# @njit(parallel=True, fastmath=True)
# @njit


@njit
def relu_1d_derivative(inputs):
    return np.array([1. if x > 0 else 0. for x in inputs])

@njit
def relu_2d_derivative(inputs):
    output = np.empty(inputs.shape)
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            output[i, j] = 1. if inputs[i, j] > 0 else 0.
    return output


@njit
def relu_3d_derivative(inputs):
    output = np.empty(inputs.shape)
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            for k in range(inputs.shape[2]):
                output[i, j, k] = 1. if inputs[i, j, k] > 0 else 0.
    return output

# def relu_derivative(inputs):
    # return np.array([1 if x > 0 else 0 for x in inputs.flat])


# @njit(parallel=True, fastmath=True)
# @njit
# def relu(inputs):
#     return np.array([max(0., x) for x in inputs.flat])


@njit
def relu_1d(inputs):
    return (np.array([max(0., x) for x in inputs]))

@njit
def relu_2d(inputs):
    output = np.empty(inputs.shape)
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            output[i, j] = max(0., inputs[i, j])
    return output


@njit
def relu_3d(inputs):
    output = np.empty(inputs.shape)
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            for k in range(inputs.shape[2]):
                output[i, j, k] = max(0., inputs[i, j, k])
    return output


class Relu(Activation):

    def __init__(self, max_value=None, negative_slope=0, threshold=0):
        self.max_value = max_value
        self.negative_slope = negative_slope
        self.threshold = threshold

    def __call__(self, inputs):
        if inputs.ndim == 1:
            return relu_1d(inputs)
        elif inputs.ndim == 2:
            return relu_2d(inputs)
        elif inputs.ndim == 3:
            return relu_3d(inputs)

    def derivative(self, inputs):
        if inputs.ndim == 1:
            return relu_1d_derivative(inputs)
        elif inputs.ndim == 2:
            return relu_2d_derivative(inputs)
        elif inputs.ndim == 3:
            return relu_3d_derivative(inputs)
