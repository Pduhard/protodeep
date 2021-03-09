
from Protodeep.initializers.HeNormal import HeNormal
from Protodeep.initializers.GlorotNormal import GlorotNormal
from Protodeep.initializers.RandomNormal import RandomNormal
from Protodeep.initializers.Zeros import Zeros
from Protodeep.activations.Relu import Relu
from Protodeep.activations.Softmax import Softmax
from Protodeep.activations.Sigmoid import Sigmoid
from Protodeep.optimizers.SGD import SGD
from Protodeep.optimizers.Adagrad import Adagrad
from Protodeep.optimizers.RMSProp import RMSProp
from Protodeep.optimizers.Adadelta import Adadelta
from Protodeep.optimizers.Adam import Adam
from Protodeep.metrics.Accuracy import Accuracy
from Protodeep.losses.BinaryCrossentropy import BinaryCrossentropy


def parse_metrics(metrics):
    res = []
    for metric in metrics:
        metric = metric.lower()
        if isinstance(metric, str) is False:
            res.append(metric)
        else:
            if metric == "accuracy":
                res.append(Accuracy())
    return res


def parse_initializer(initializer):
    if isinstance(initializer, str) is False:
        return initializer
    initializer = initializer.lower()
    if initializer == "henormal":
        return HeNormal()
    elif initializer == "glorotnormal":
        return GlorotNormal()
    elif initializer == "randomnormal":
        return RandomNormal()
    elif initializer == "zeros":
        return Zeros()
    else:
        return HeNormal()


def parse_optimizer(optimizer):
    if isinstance(optimizer, str) is False:
        return optimizer
    optimizer = optimizer.lower()
    if optimizer == "sgd":
        return SGD()
    elif optimizer == "adagrad":
        return Adagrad()
    elif optimizer == "rmsprop":
        return RMSProp()
    elif optimizer == "adam":
        return Adam()
    elif optimizer == "adadelta":
        return Adadelta()
    else:
        return Adam()


def parse_loss(loss):
    if isinstance(loss, str) is False:
        return loss
    loss = loss.lower()
    if loss == "binarycrossentropy" or loss == "binary_crossentropy":
        return BinaryCrossentropy()
    else:
        return BinaryCrossentropy()


# def parse_initializer(initializer):
#     if isinstance(initializer, str) is False:
#         return initializer
#     initializer = initializer.lower()
#     if initializer == "henormal":
#         return HeNormal()
#     elif initializer == "glorotnormal":
#         return GlorotNormal()
#     elif initializer == "randomnormal":
#         return RandomNormal()
#     else:
#         return HeNormal()


def parse_activation(activation):
    if isinstance(activation, str) is False:
        return activation
    activation = activation.lower()
    if activation == "softmax":
        return Softmax()
    elif activation == "sigmoid":
        return Sigmoid()
    elif activation == "relu":
        return Relu()
    else:
        return Relu()  # !! linear