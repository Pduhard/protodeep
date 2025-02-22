from dataset import Dataset
from Protodeep.model.model import Model
from Protodeep.layers.Dense import Dense
from Protodeep.layers.Input import Input
import sys
import matplotlib.pyplot as plt
from Protodeep.callbacks.EarlyStopping import EarlyStopping
from os import path

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
        "epoch": parse_option_value("-e", dflt="100")
        }
    if check_option(options) is False:
        usage()
    return options

csvPath  = path.join(path.dirname(__file__), 'data.csv')

if __name__ == "__main__":
    options = parse_options()
    dataset = Dataset(csvPath, 0.2)
    i = Input((30))()
    i2 = Input((30))()
    d1 = Dense(64, activation="relu")(i)
    d1bis = Dense(64, activation="relu")(i2)
    d2 = Dense(32, activation="relu")(d1)
    d2bis = Dense(32, activation="relu")(d1bis)
    out1 = Dense(2, activation="softmax")(d2)
    
    out2 = Dense(2, activation="softmax")(d2bis)

    d3 = Dense(128, activation="relu")(i)
    out3 = Dense(2, activation="softmax")(d3)

    model = Model(inputs=[i, i2], outputs=[out1, out2, out3])
    model.compile(30, metrics=["accuracy"], optimizer="Adam")
    print(dataset.features.shape)
    print(dataset.test_features.shape)

    model.summary()
    history = model.fit(
        features=[dataset.features, dataset.features],
        targets=[dataset.targets, dataset.targets, dataset.targets],
        epochs=500,
        batch_size=32,
        validation_data=([dataset.test_features, dataset.test_features], [dataset.test_targets, dataset.test_targets, dataset.test_targets]),
        callbacks=[EarlyStopping(monitor="val_loss", patience=15)]
    )
    model.evaluate(
        validation_data=([dataset.test_features, dataset.test_features], [dataset.test_targets, dataset.test_targets, dataset.test_targets])
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
