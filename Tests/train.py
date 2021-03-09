from dataset import Dataset
from Protodeep.model.model import Model
# from layers.Dense import Dense 
# from layers.Dense import Dense 
from Protodeep.layers.Dense import Dense
import sys
# import numpy as np
import matplotlib.pyplot as plt
from Protodeep.callbacks.EarlyStopping import EarlyStopping


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
        'optimizer': parse_option_value('-o', dflt=None),
        'epoch': parse_option_value('-e', dflt='100'),
        'csv_name': parse_option_value('-n', dflt='data.csv')
        }
    if check_option(options) is False:
        usage()
    return options


if __name__ == "__main__":
    options = parse_options()
    # print(options)
    # dataset = Dataset("../data_training.csv")
    # dataset_test = Dataset("../data_test.csv")
    dataset = Dataset(options['csv_name'], 0.2)
    model = Model()
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    # model.add(64, activation="relu")
    # model.add(32, activation="relu")
    # model.add(2, activation="softmax")
    model.compile(30, metrics=["accuracy"], optimizer="Adam")
    print(dataset.features.shape)
    print(dataset.test_features.shape)
    # print(dataset.features.shape)
    # print(numpy.min(model.weights[0]))
    history = model.fit(
        features=dataset.features,
        targets=dataset.targets,
        epochs=500,
        batch_size=32,
        validation_data=(dataset.test_features, dataset.test_targets)
        ,callbacks=[EarlyStopping(monitor="val_loss", patience=6)]
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
    # print("val_loss", val_loss)
    # e = np.array([1, 2, 3])
    # test(e)
    # print(e)
