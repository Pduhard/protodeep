import numpy as np


class Adagrad:

    def __init__(self, learning_rate=0.01,
                 initial_acc_value=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.initial_acc_value = initial_acc_value
        self.epsilon = epsilon
        
        self.acc = []
        self.apply_gradient = None


        self.apply_gradient = self.adagrad

    def add_weight(self, weight):
        self.acc.append(np.full(weight.shape, self.initial_acc_value))

    def compile(self, model):
        for weight in model.weights:
            self.add_weight(weight)

    def adagrad(self, weights, gradients):
        lr = self.learning_rate
        e = self.epsilon
        for i, (weight, gradient) in enumerate(zip(weights, gradients)):
            self.acc[i] = self.acc[i] + gradient ** 2
            weight -= lr / np.sqrt(self.acc[i] + e) * gradient

    def __str__(self):
        return "Adagrad:\n\t\
                learning rate= {}\t\
                initial_acc_value= {}\t\
                epsilon= {}".format(self.learning_rate,
                                    self.initial_acc_valule,
                                    self.epsilon)
