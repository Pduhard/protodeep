import numpy as np


class Accuracy:

    def __init__(self, name='accuracy'):
        self.name = name
        self.count = 0
        self.total = 0

    def reset_state(self):
        self.count = 0
        self.total = 0

    def result(self):
        if self.total == 0:
            return 0
        return self.count / self.total

    def update_state(self, prediction, target):
        # print(prediction[0].shape, target)
        # quit()

        for i in range(len(prediction)):
            self.total += prediction[i].shape[0]
            self.count += np.sum(np.argmax(prediction[i], axis=-1) == np.argmax(target[i], axis=-1))
                # self.count += 1
        # if isinstance(prediction, list):
        #     for i in range(len(prediction)):
        #         self.total += 1
        #         if np.argmax(prediction[0]) == np.argmax(target[0]):
        #             self.count += 1
        # else:
        #     self.total += 1
        #     if np.argmax(prediction) == np.argmax(target):
        #         self.count += 1
