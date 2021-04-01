
from Protodeep.initializers.HeNormal import HeNormal
from Protodeep.initializers.GlorotNormal import GlorotNormal
from Protodeep.initializers.RandomNormal import RandomNormal
from Protodeep.initializers.Zeros import Zeros
from Protodeep.activations.Relu import Relu
from Protodeep.activations.Softmax import Softmax
from Protodeep.activations.Sigmoid import Sigmoid
from Protodeep.activations.Linear import Linear
from Protodeep.activations.Tanh import Tanh
from Protodeep.optimizers.SGD import SGD
from Protodeep.optimizers.Adagrad import Adagrad
from Protodeep.optimizers.RMSProp import RMSProp
from Protodeep.optimizers.Adadelta import Adadelta
from Protodeep.optimizers.Adam import Adam
from Protodeep.metrics.Accuracy import Accuracy
from Protodeep.losses.BinaryCrossentropy import BinaryCrossentropy
from Protodeep.losses.MeanSquaredError import MeanSquaredError
from Protodeep.regularizers.L1 import L1
from Protodeep.regularizers.L2 import L2
from Protodeep.regularizers.L1L2 import L1L2


def parse_metrics(metrics):
    res = []
    for metric in metrics:
        if isinstance(metric, str) is False:
            res.append(metric)
        else:
            metric = metric.lower().replace('_', '')
            if metric == "accuracy":
                res.append(Accuracy())
    return res


def parse_initializer(initializer):
    if isinstance(initializer, str) is False:
        return initializer or HeNormal()
    initializer = initializer.lower().replace('_', '')
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
        return optimizer or Adam()
    optimizer = optimizer.lower().replace('_', '')
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
        return loss or BinaryCrossentropy()
    loss = loss.lower().replace('_', '')
    if loss == "binarycrossentropy":
        return BinaryCrossentropy()
    elif loss == "meansquarederror":
        return MeanSquaredError()
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
        return activation or Linear()
    activation = activation.lower()
    if activation == "softmax":
        return Softmax()
    elif activation == "sigmoid":
        return Sigmoid()
    elif activation == "relu":
        return Relu()
    elif activation == "tanh":
        return Tanh()
    elif activation == "linear":
        return Linear()
    else:
        return Linear()  # !! linear

        
def parse_regularizer(regularizer):
    if isinstance(regularizer, str) is False:
        return regularizer or None
    regularizer = regularizer.lower()
    if regularizer == "l1":
        return L1()
    elif regularizer == "l2":
        return L2()
    elif regularizer == "l1l2":
        return L1L2()
    else:
        return None