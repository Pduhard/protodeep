from dataset import Dataset
from Protodeep.model.model import Model
from Protodeep.layers.Dense import Dense
from Protodeep.layers.Conv2D import Conv2D
from Protodeep.layers.MaxPool2D import MaxPool2D
from Protodeep.layers.Flatten import Flatten
from Protodeep.layers.Input import Input
import sys
# import numpy as np
import matplotlib.pyplot as plt
from Protodeep.callbacks.EarlyStopping import EarlyStopping
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
        'csv_name': parse_option_value('-n', dflt='mnist_784.csv')
        }
    if check_option(options) is False:
        usage()
    return options


# TENSORFLOW 

import numpy as np
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf


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
        # quit()
        model = tf.keras.models.Sequential()

        # quit()
        model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 2), activation='relu', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPool2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 2), activation='relu'))
        model.add(tf.keras.layers.MaxPool2D((2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        model.compile(
            # optimizer=tf.keras.optimizers.SGD(
            #     learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD'),
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"]
        )
        model.summary()
        print(model.layers[0].get_weights()[0].shape)
        # print(self.targets.shape)
        # print(self.features.shape)
        # quit()
        # print(self.features[0:1].shape)
        # quit()
        # model_output = model.predict(self.features[0:1])
        # print(model_output, self.targets[0:1])
        return model


    def __init__(self, file_name=None):
        (self.features, self.targets), (self.test_features, self.test_targets) = parse_mnist()
        # self.features 
        print(self.features[0].shape)
        # scaler = StandardScaler()
        # self.features = scaler.fit_transform(self.features)
        # self.test_features = scaler.fit_transform(self.test_features)
        # if file_name == "../data_training.csv":
        model = self.get_model_conv()
        history = model.fit(self.features, self.targets, epochs=25, validation_data=(self.test_features, self.test_targets)
            , callbacks=[tf.keras.callbacks.EarlyStopping()])
        # model.summary()
        # history = model.fit(self.features, self.targets, epochs=50, batch_size=569)
        # loss_curve = history.history["loss"]
        # acc_curve = history.history["accuracy"]
        # plt.plot(loss_curve, label="Train")
        # plt.legend(loc='upper left')
        # plt.title("Loss")
        # plt.show()
        # plt.plot(acc_curve, label="Train")
        # plt.legend(loc='upper left')
        # plt.title("Accuracy")
        # plt.show()
        # loss, acc = model.evaluate(self.features, self.targets)
        # print("Test Loss", loss)
        # print("Test Accuracy", acc)


from Protodeep.utils.debug import class_timer


# @class_timer
# def test(a, b):
#     return (a + b)

if __name__ == "__main__":
    options = parse_options()
    dataset = Dataset(options['csv_name'], 0.2)

    # test(1, 2)
    # s = Dttf()
    i = Input((28, 28, 1))()
    # with Conv2D(filters=2, kernel_size=(3, 3), activation='relu') as a 
    a = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu')
    conv1 = a(i)
    maxpool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation='relu')(maxpool1)
    maxpool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    flatten = Flatten()(maxpool2)
    # flatten = Flatten()(conv1)
    d1 = Dense(64, activation="relu")(flatten)
    d2 = Dense(32, activation="relu")(d1)
    output = Dense(10, activation="softmax")(d2)
    # output2 = Dense(10, activation="softmax")(d2)
    # model.add(MaxPooling2D((2, 2)))
    model = Model(inputs=i, outputs=output)
    # model.add(Flatten())
    # model.add(Dense(32, activation="relu"))
    # model.add(Dense(16, activation="relu"))
    # model.add(Dense(10, activation="softmax"))
    # model.add(64, activation="relu")
    # model.add(32, activation="relu")
    # model.add(2, activation="softmax")
    model.compile((28, 28, 1), metrics=["accuracy"], optimizer="Adam")
    
    model.summary()
    from Protodeep.layers.Layer import Layer

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
        callbacks=[EarlyStopping(monitor="val_loss", patience=2)]
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

    for i in range(10):
        print(np.argmax(model.predict(dataset.test_features[i:i+1])), dataset.test_targets[i:i+1])
        plt.imshow(dataset.test_features[i], cmap='gray')
        plt.show()
    # print("val_loss", val_loss)
    # e = np.array([1, 2, 3])
    # test(e)
    # print(e)

# if __name__ == "__main__":
#     options = parse_options()
#     dataset = Dataset("mnist_784.csv", 0.2)
#     model = Model()

#     # test(1, 2)
#     # s = Dttf()
#     model.add(Conv2D(filters=2, kernel_size=(3, 3), activation='relu'))
#     model.add(MaxPool2D(pool_size=(2, 2)))
#     model.add(Conv2D(filters=2, kernel_size=(3, 3), activation='relu'))
#     # model.add(MaxPool2D(pool_size=(2, 2)))
#     # model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
#     model.add(MaxPool2D(pool_size=(2, 2)))
#     # model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
#     # model.add(MaxPooling2D((2, 2)))
#     model.add(Flatten())
#     model.add(Dense(32, activation="relu"))
#     model.add(Dense(16, activation="relu"))
#     model.add(Dense(10, activation="softmax"))
#     # model.add(64, activation="relu")
#     # model.add(32, activation="relu")
#     # model.add(2, activation="softmax")
#     model.compile((28, 28, 1), metrics=["accuracy"], optimizer="Adam")
    
#     model.summary()
#     from layers.Layer import Layer

#     Layer.print_dico()
#     for name, layer in Layer.layer_dico.items():
#         print(name)
#         print(layer.input_connectors)
#         print(layer.output_connectors)
#     # quit()
#     # quit()
#     # print(dataset.features.shape)
#     # print(dataset.test_features.shape)
#     # print(dataset.features.shape)
#     # print(numpy.min(model.weights[0]))
#     history = model.fit(
#         features=dataset.features,
#         targets=dataset.targets,
#         epochs=10,
#         batch_size=32,
#         validation_data=(dataset.test_features, dataset.test_targets),
#         callbacks=[EarlyStopping(monitor="val_loss", patience=6)]
#     )
#     model.evaluate(
#         validation_data=(dataset.test_features, dataset.test_targets)
#     )
#     plt.plot(history['accuracy'])
#     plt.plot(history['val_accuracy'])
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()

#     plt.plot(history['loss'])
#     plt.plot(history['val_loss'])
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()
#     # print("val_loss", val_loss)
#     # e = np.array([1, 2, 3])
#     # test(e)
#     # print(e)
