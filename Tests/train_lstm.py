from dataset import Dataset
from Protodeep.model.model import Model
from Protodeep.layers.Dense import Dense
from Protodeep.layers.Conv2D import Conv2D
from Protodeep.layers.MaxPool2D import MaxPool2D
from Protodeep.layers.LSTM import LSTM
from Protodeep.layers.Flatten import Flatten
from Protodeep.layers.Input import Input
import sys
# import numpy as np
import matplotlib.pyplot as plt
from Protodeep.callbacks.EarlyStopping import EarlyStopping
from Protodeep.optimizers.SGD import SGD
# from ..dataset_tf import Dataset as Dttf


def parse_option_value(opt, dflt):
    if opt in sys.argv:
        if sys.argv.index(opt) + 1 != len(sys.argv):
            return sys.argv[sys.argv.index(opt) + 1]
    return dflt


def usage():
    print("usage : blabla")
    quit()


def check_option(options):
    return True


def parse_options():
    options = {
        "optimizer": parse_option_value("-o", dflt=None),
        "epoch": parse_option_value("-e", dflt="100"),
        'csv_name': parse_option_value('-n', dflt='BTCUSD_day.csv')
        }
    if check_option(options) is False:
        usage()
    return options


# TENSORFLOW 

import numpy as np
import matplotlib.pyplot as plt

from Protodeep.utils.debug import class_timer


# @class_timer
# def test(a, b):
#     return (a + b)

if __name__ == "__main__":
    options = parse_options()
    dataset = Dataset(options['csv_name'], 0.2)

    print(dataset.features.shape, dataset.targets.shape)
    # test(1, 2)
    # s = Dttf()
    i = Input((10, 4))()
    lstm = LSTM(32)(i)
    output = Dense(1)(lstm)
    model = Model(inputs=i, outputs=output)
    model.compile((10, 4), metrics=["accuracy"], optimizer='Adam')
    
    model.summary()
    # from Protodeep.layers.Layer import Layer

    # Layer.print_dico()
    # for name, layer in Layer.layer_dico.items():
    #     print(name)
    #     print(layer.input_connectors)
    #     print(layer.output_connectors)
    # quit()
    # quit()
    # print(dataset.features.shape)
    # print(dataset.test_features.shape)
    # print(dataset.features.shape)
    # print(numpy.min(model.weights[0]))
    print(dataset.features.shape)
    history = model.fit(
        features=dataset.features,
        targets=dataset.targets,
        epochs=100,
        batch_size=32,
        validation_data=(dataset.test_features, dataset.test_targets),
        callbacks=[EarlyStopping(monitor="val_loss", patience=3)]
    )
    model.evaluate(
        validation_data=(dataset.test_features, dataset.test_targets)
    )
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
