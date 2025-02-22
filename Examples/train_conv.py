from dataset import Dataset
from Protodeep.model.model import Model
from Protodeep.layers.Dense import Dense
from Protodeep.layers.Conv2D import Conv2D
from Protodeep.layers.MaxPool2D import MaxPool2D
from Protodeep.layers.Flatten import Flatten
from Protodeep.layers.Input import Input
import sys
import matplotlib.pyplot as plt
from Protodeep.callbacks.EarlyStopping import EarlyStopping
from Protodeep.optimizers.SGD import SGD

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
        'csv_name': parse_option_value('-n', dflt='mnist_784.csv')
        }
    if check_option(options) is False:
        usage()
    return options



import numpy as np
import matplotlib.pyplot as plt

def parse_mnist():
    (train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.mnist.load_data()
    test_X = test_X[:1000].reshape((1000, 28, 28, 1))
    train_X = train_X[:1000].reshape((1000, 28, 28, 1))
    test_Y = np.eye(10)[test_Y[:1000]]
    train_Y = np.eye(10)[train_Y[:1000]]
    print('ee')
    return (train_X.astype(float) / 255., train_Y.astype(float)), (test_X.astype(float) / 255., test_Y.astype(float))


class Dttf:

    def get_model_conv(self):
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 2), activation='relu', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPool2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 2), activation='relu'))
        model.add(tf.keras.layers.MaxPool2D((2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"]
        )
        model.summary()
        print(model.layers[0].get_weights()[0].shape)
        return model


    def __init__(self, file_name=None):
        (self.features, self.targets), (self.test_features, self.test_targets) = parse_mnist()
        print(self.features[0].shape)
        model = self.get_model_conv()
        history = model.fit(self.features, self.targets, epochs=25, validation_data=(self.test_features, self.test_targets)
            , callbacks=[tf.keras.callbacks.EarlyStopping()])


from Protodeep.utils.debug import class_timer

def get_light_config():
    i = Input((28, 28, 1))()
    a = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', use_bias=False)
    conv1 = a(i)
    maxpool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', use_bias=True)(maxpool1)
    maxpool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    flatten = Flatten()(maxpool2)
    d1 = Dense(64, activation="relu", use_bias=False)(flatten)
    d2 = Dense(32, activation="relu", use_bias=False)(d1)
    output = Dense(10, activation="softmax")(d2)
    model = Model(inputs=i, outputs=output)
    model.compile((28, 28, 1), metrics=["categorical_accuracy"], optimizer='Adam')
    model.summary()
    return model
def get_heavy_config():
    i = Input((28, 28, 1))()
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu')(i)
    maxpool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(maxpool1)
    maxpool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(maxpool2)
    flatten = Flatten()(conv3)
    d1 = Dense(128, activation="relu")(flatten)
    d2 = Dense(64, activation="relu")(d1)
    output = Dense(10, activation="softmax")(d2)
    model = Model(inputs=i, outputs=output)
    model.compile((28, 28, 1), metrics=["categorical_accuracy"], optimizer='Adam')
    
    model.summary()
    return model

if __name__ == "__main__":
    options = parse_options()
    dataset = Dataset(options['csv_name'], 0.2)

    model = get_light_config()
    from Protodeep.layers.Layer import Layer

    print(dataset.features.shape)
    history = model.fit(
        features=dataset.features,
        targets=dataset.targets,
        epochs=10,
        batch_size=32,
        validation_data=(dataset.test_features, dataset.test_targets),
        callbacks=[EarlyStopping(monitor="val_loss", patience=2)]
    )
    model.evaluate(
        validation_data=(dataset.test_features, dataset.test_targets)
    )
    plt.plot(history['categorical_accuracy'])
    plt.plot(history['val_categorical_accuracy'])
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

    for i in range(10):
        print(np.argmax(model.predict(dataset.test_features[i:i+1])), dataset.test_targets[i:i+1])
        plt.imshow(dataset.test_features[i].reshape(28, 28), cmap='gray')
        plt.show()


    


