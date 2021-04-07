import numpy as np


class RMSProp:

    def __init__(self, learning_rate=0.001, rho=0.9,
                 momentum=0.0, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon

        self.acc = []
        self.velocity = []
        self.apply_gradient = None

        if momentum > 0:
            self.apply_gradient = self.momentum_rmsprop
        else:
            self.apply_gradient = self.rmsprop

    def add_weight(self, weight):
        self.acc.append(np.zeros(weight.shape))
        if self.momentum > 0:
            self.velocity.append(np.zeros(weight.shape))

    def compile(self, model):
        for weight in model.weights:
            self.add_weight(weight)

    def momentum_rmsprop(self, weights, gradients):
        rho = self.rho
        lr = self.learning_rate
        m = self.momentum
        e = self.epsilon
        for i, (weight, gradient) in enumerate(zip(weights, gradients)):
            self.acc[i] = rho * self.acc[i] + (1 - rho) * gradient ** 2
            vel_add = lr * gradient / np.sqrt(self.acc[i] + e)
            self.velocity[i] = self.velocity[i] * m + vel_add
            weight -= self.velocity[i]

    def rmsprop(self, weights, gradients):
        rho = self.rho
        lr = self.learning_rate
        e = self.epsilon
        for i, (weight, gradient) in enumerate(zip(weights, gradients)):
            self.acc[i] = rho * self.acc[i] + (1 - rho) * gradient ** 2
            weight -= lr * gradient / np.sqrt(self.acc[i] + e)

    def __str__(self):
        return 'RMSProp:\n\t\
            learning rate= {}\t\
            momentum= {}\t\
            rho= {}'.format(self.learning_rate,
                            self.momentum,
                            self.rho)
