import numpy as np


class SGD:

    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.velocity = []
        self.apply_gradient = None

        if self.momentum <= 0:
            self.apply_gradient = self.stochastic_gradient_descent
        else:
            self.apply_gradient = self.momentum_sgd

    def add_weight(self, weight):
        self.velocity.append(np.zeros(weight.shape))

    def compile(self, model):
        if self.momentum > 0:
            for weight in model.weights:
                self.add_weight(weight)

    def stochastic_gradient_descent(self, weights, gradients):
        for i, (weight, gradient) in enumerate(zip(weights, gradients)):
            weight -= self.learning_rate * gradient

    def momentum_sgd(self, weights, gradients):
        lr = self.learning_rate
        m = self.momentum
        for i, (weight, gradient) in enumerate(zip(weights, gradients)):
            self.velocity[i] = self.velocity[i] * m - lr * gradient
            weight += self.velocity[i]

    def __str__(self):
        return 'SGD:\n\t\
            learning rate= {}\t\
            momentum= {}'.format(self.learning_rate,
                                 self.momentum)
