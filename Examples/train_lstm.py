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

from Scalers.StandardScaler import StandardScaler
from Preprocessing.Split import Split

def parse_option_value(opt, dflt):
    if opt in sys.argv and sys.argv.index(opt) + 1 != len(sys.argv):
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

    scaler = StandardScaler().fit(dataset.features)
    dataset.features = scaler.transform(dataset.features)

    trg_scaler = StandardScaler().fit(dataset.targets)
    dataset.targets = trg_scaler.transform(dataset.targets)

    features, targets = Split.time_series_split(dataset.features, dataset.targets, ssize=10)

    print(features[0].tolist())
    print(targets[0].tolist())

    ((x_train, y_train), (x_test, y_test)) = Split.train_test_split(
        features, targets)
    # print(features.shape, dataset.targets.shape)
    # test(1, 2)
    # s = Dttf()
    i = Input((10, 4))()
    lstm = LSTM(32, return_sequences=True, kernel_regularizer='l1l2', use_bias=False)(i)
    lstm1 = LSTM(32, use_bias=False)(lstm)
    output = Dense(1)(lstm1)
    model = Model(inputs=i, outputs=output)
    model.compile(
        (10, 4),
        metrics=[],
        optimizer='Adam',
        loss='mean_squared_error'
    )
    
    model.summary()
    # from Protodeep.layers.Layer import Layer

    # Layer.print_dico()
    # for name, layer in Layer.layer_dico.items():
    #     print(name)
    #     print(layer.input_connectors)
    #     print(layer.output_connectors)
    # quit()
    # quit()
    # print(features.shape)
    # print(dataset.test_features.shape)
    # print(features.shape)
    # print(numpy.min(model.weights[0]))
    print(features.shape)
    history = model.fit(
        features=x_train,
        targets=y_train,
        epochs=100,
        batch_size=32,
        validation_data=(x_test, y_test),
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
    )
    print(model.evaluate(
        validation_data=(x_test, y_test)
    ))
    # plt.plot(history['accuracy'])
    # plt.plot(history['val_accuracy'])
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    for i in range(20):
        print(model.predict(x_test[i:i+1]), y_test[i:i+1])