import numpy as np
# from numba import njit


# @njit
def test(lr, v_at, epsilon, m_at):
    return (lr / (np.sqrt(v_at) + epsilon)) * m_at


class Adam:

    def __init__(self, learning_rate=0.001, beta_1=0.9,
                 beta_2=0.999, epsilon=1e-7, amsgrad=False):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

        self.v = []
        self.m = []
        self.v_at = []

        self.apply_gradient = None
        self.t = 1

        if self.amsgrad:
            self.apply_gradient = self.amsgrad_adam
        else:
            self.apply_gradient = self.adam

    def add_weight(self, weight):
        """
            can be used outside a neural nework
        """
        self.v.append(np.zeros(weight.shape))
        self.m.append(np.zeros(weight.shape))
        if self.amsgrad:
            self.v_at.append(np.full(weight.shape, -2))

    def compile(self, model):
        """
            used after model is compilled
        """
        for weight in model.weights:
            self.add_weight(weight)

    def amsgrad_adam(self, weights, gradients):
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        e = self.epsilon
        lr = self.learning_rate
        for i, (weight, gradient) in enumerate(zip(weights, gradients)):
            self.m[i] = beta_1 * self.m[i] + (1 - beta_1) * gradient
            self.v[i] = beta_2 * self.v[i] + (1 - beta_2) * gradient ** 2
            m_at = self.m[i] / (1 - (beta_1 ** self.t))
            v_at = self.v[i] / (1 - (beta_2 ** self.t))
            self.v_at[i] = np.maximum(self.v_at[i], v_at)
            weight -= lr * m_at / (np.sqrt(self.v_at[i]) + e)
        self.t += 1

    def adam(self, weights, gradients):
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        e = self.epsilon
        lr = self.learning_rate
        for i, (weight, gradient) in enumerate(zip(weights, gradients)):
            self.m[i] = beta_1 * self.m[i] + (1 - beta_1) * gradient
            self.v[i] = beta_2 * self.v[i] + (1 - beta_2) * gradient ** 2
            m_at = self.m[i] / (1 - (beta_1 ** self.t))
            v_at = self.v[i] / (1 - (beta_2 ** self.t))
            weight -= lr * m_at / (np.sqrt(v_at) + e)
        self.t += 1
