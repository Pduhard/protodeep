# import numpy as np


class Adadelta:

    velocity = []
    delta = []
    dwb = []
    apply_gradient = None

    def __init__(self, learning_rate=0.001, rho=0.95, epsilon=1e-6):
        print("please don't use adadelta not working")
        quit()
    #     self.learning_rate = learning_rate
    #     self.rho = rho
    #     self.epsilon = epsilon
    #     self.apply_gradient = self.adagrad

    # def compile(self, model):
    #     for i in range(len(model.layers)):
    #         self.velocity.append({
    #             "w": np.zeros(model.weights[i].shape),
    #             "b": np.zeros(model.biases[i].shape)
    #         })
    #         self.delta.append({
    #             "w": np.zeros(model.weights[i].shape),
    #             "b": np.zeros(model.biases[i].shape)
    #         })
    #         self.dwb.append({
    #             "w": np.zeros(model.weights[i].shape),
    #             "b": np.zeros(model.biases[i].shape)
    #         })

    # # pas fou :'(
    # def adagrad(self, model, batch_size):
    #     for i in range(len(model.layers)):
    #         # old delta =
    #         self.delta[i]["w"] = self.rho * self.delta[i]["w"]
    #         + (1 - self.rho) * (self.dwb[i]["w"] ** 2)
    #         self.delta[i]["b"] = self.rho * self.delta[i]["b"]
    #         + (1 - self.rho) * (self.dwb[i]["b"] ** 2)

    #         self.velocity[i]["w"] = self.rho * self.velocity[i]["w"]
    #         + (1 - self.rho) * ((model.delta_weights[i] / batch_size) ** 2)
    #         self.velocity[i]["b"] = self.rho * self.velocity[i]["b"]
    #         + (1 - self.rho) * ((model.delta_biases[i] / batch_size) ** 2)

    #         # self.acc[i]["b"] = self.rho * self.acc[i]["b"] + (1 - self.rho)
    #         # * ((model.delta_biases[i] / batch_size) ** 2)
    #         new_w = model.weights[i]
    #         - (np.sqrt(self.delta[i]["w"] + self.epsilon) /
    #             np.sqrt(self.velocity[i]["w"] + self.epsilon))
    #                 * (model.delta_weights[i] / batch_size)
    #         new_b = model.biases[i]
    #         - (np.sqrt(self.delta[i]["b"] + self.epsilon)
    #             / np.sqrt(self.velocity[i]["b"] + self.epsilon))
    #                 * (model.delta_biases[i] / batch_size)
    #         # self.dwb[i]["w"] = new_w - model.delta_weights[i]
    #         # self.dwb[i]["b"] = new_b - model.delta_biases[i]
    #         model.weights[i] = new_w
    #         model.biases[i] = new_b
