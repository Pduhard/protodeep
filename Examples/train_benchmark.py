import sys

import matplotlib.pyplot as plt
import numpy as np

import Protodeep as P
from dataset import Dataset


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
        'optimizer': parse_option_value('-o', dflt=None),
        'epoch': parse_option_value('-e', dflt='100'),
        'csv_name': parse_option_value('-n', dflt='data.csv')
        }
    if check_option(options) is False:
        usage()
    return options


def get_model_regularizers():
    model = P.model.Model()
    i = P.layers.Input(shape=(30))()
    d1 = P.layers.Dense(64, activation=P.activations.Relu(), kernel_regularizer=P.regularizers.L1L2())(i)
    d2 = P.layers.Dense(32, activation=P.activations.Relu(), kernel_regularizer='')(d1)
    out = P.layers.Dense(2, activation=P.activations.Softmax())(d2)

    model = P.model.Model(inputs=i, outputs=out)
    model.compile(30, metrics=[P.metrics.Accuracy()], optimizer=P.optimizers.Adam())
    model.summary()
    return model

def get_basic_model():
    i = P.layers.Input(shape=(30))()
    d1 = P.layers.Dense(64, activation=P.activations.Relu())(i)
    d2 = P.layers.Dense(32, activation=P.activations.Relu())(d1)
    out = P.layers.Dense(2, activation=P.activations.Softmax())(d2)

    model = P.model.Model(inputs=i, outputs=out)

    model.compile(30, metrics=[P.metrics.Accuracy()], optimizer=P.optimizers.Adam())
    model.summary()
    return model

def get_basic_model_RMSProp():
    i = P.layers.Input(shape=(30))()
    d1 = P.layers.Dense(64, activation=P.activations.Relu())(i)
    d2 = P.layers.Dense(32, activation=P.activations.Relu())(d1)
    out = P.layers.Dense(2, activation=P.activations.Softmax())(d2)

    model = P.model.Model(inputs=i, outputs=out)

    model.compile(30, metrics=[P.metrics.Accuracy()], optimizer=P.optimizers.Adam())
    model.summary()
    return model


def get_troll_model_for_bonuses():
    model = P.model.Model()
    model.add(P.layers.Conv2D(filters=30, kernel_size=(2, 2), activation=P.activations.Relu()))
    model.add(P.layers.Flatten())
    model.add(P.layers.Dense(32, activation=P.activations.Relu()))
    model.add(P.layers.Dense(2, activation=P.activations.Softmax()))
    model.compile((5, 6, 1), metrics=[P.metrics.Accuracy()], optimizer=P.optimizers.RMSProp())
    model.summary()
    return model

if __name__ == "__main__":
    options = parse_options()
    dataset = Dataset(options['csv_name'], 0.2)
    modelreg = get_model_regularizers()
    model_simple = get_basic_model()
    model_rms = get_basic_model_RMSProp()

    print(dataset.features.shape)
    print(dataset.test_features.shape)
    history_reg = modelreg.fit(
        features=dataset.features,
        targets=dataset.targets,
        epochs=500,
        batch_size=32,
        validation_data=(dataset.test_features, dataset.test_targets),
        callbacks=[P.callbacks.EarlyStopping(monitor="val_loss", patience=3, baseline=0.029, restore_best_weights=True)],
        verbose=False
    )
    print('model reg')
    print(modelreg.evaluate(
        validation_data=(dataset.test_features, dataset.test_targets)
    ))

    history_simple = model_simple.fit(
        features=dataset.features,
        targets=dataset.targets,
        epochs=500,
        batch_size=32,
        validation_data=(dataset.test_features, dataset.test_targets),
        callbacks=[P.callbacks.EarlyStopping(monitor="val_loss", patience=10)],
        verbose=False
    )
    print('model simple')
    print(model_simple.evaluate(
        validation_data=(dataset.test_features, dataset.test_targets)
    ))

    history_rms = model_rms.fit(
        features=dataset.features,
        targets=dataset.targets,
        epochs=500,
        batch_size=32,
        validation_data=(dataset.test_features, dataset.test_targets),
        callbacks=[P.callbacks.EarlyStopping(monitor="val_loss", patience=10)],
        verbose=True
    )
    print('model rms')
    print(model_rms.evaluate(
        validation_data=(dataset.test_features, dataset.test_targets)
    ))
    plt.plot(history_simple['accuracy'])
    plt.plot(history_simple['val_accuracy'])
    plt.plot(history_reg['accuracy'])
    plt.plot(history_reg['val_accuracy'])
    plt.plot(history_rms['accuracy'])
    plt.plot(history_rms['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history_simple['loss'])
    plt.plot(history_simple['val_loss'])
    plt.plot(history_reg['loss'])
    plt.plot(history_reg['val_loss'])
    plt.plot(history_rms['loss'])
    plt.plot(history_rms['val_loss'])
    
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


