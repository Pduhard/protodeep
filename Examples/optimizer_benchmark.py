import sys
import matplotlib.pyplot as plt
import numpy as np
from os import path
import Protodeep as P
from dataset import Dataset

def get_model_Adam():
    model = P.model.Model()
    i = P.layers.Input(shape=(30))()
    d1 = P.layers.Dense(64, activation='relu')(i)
    d2 = P.layers.Dense(32, activation='relu')(d1)
    out = P.layers.Dense(2, activation='softmax')(d2)

    model = P.model.Model(inputs=i, outputs=out)
    model.compile(30, metrics=['Accuracy'], optimizer='Adam')
    model.summary()
    return model

def get_model_Adam_amsgrad():
    model = P.model.Model()
    i = P.layers.Input(shape=(30))()
    d1 = P.layers.Dense(64, activation='relu')(i)
    d2 = P.layers.Dense(32, activation='relu')(d1)
    out = P.layers.Dense(2, activation='softmax')(d2)

    model = P.model.Model(inputs=i, outputs=out)
    model.compile(30, metrics=['Accuracy'], optimizer=P.optimizers.Adam(amsgrad=True))
    model.summary()
    return model


def get_model_Adagrad():
    model = P.model.Model()
    i = P.layers.Input(shape=(30))()
    d1 = P.layers.Dense(64, activation='relu')(i)
    d2 = P.layers.Dense(32, activation='relu')(d1)
    out = P.layers.Dense(2, activation='softmax')(d2)

    model = P.model.Model(inputs=i, outputs=out)
    model.compile(30, metrics=['Accuracy'], optimizer='Adagrad')
    model.summary()
    return model


def get_model_SGD():
    i = P.layers.Input(shape=(30))()
    d1 = P.layers.Dense(64, activation='relu')(i)
    d2 = P.layers.Dense(32, activation='relu')(d1)
    out = P.layers.Dense(2, activation='softmax')(d2)

    model = P.model.Model(inputs=i, outputs=out)

    model.compile(30, metrics=['Accuracy'], optimizer='SGD')
    model.summary()
    return model

def get_model_SGD_momentum():
    i = P.layers.Input(shape=(30))()
    d1 = P.layers.Dense(64, activation='relu')(i)
    d2 = P.layers.Dense(32, activation='relu')(d1)
    out = P.layers.Dense(2, activation='softmax')(d2)

    model = P.model.Model(inputs=i, outputs=out)

    model.compile(30, metrics=['Accuracy'], optimizer=P.optimizers.SGD(momentum=0.9))
    model.summary()
    return model

def get_model_RMSProp():
    i = P.layers.Input(shape=(30))()
    d1 = P.layers.Dense(64, activation='relu')(i)
    d2 = P.layers.Dense(32, activation='relu')(d1)
    out = P.layers.Dense(2, activation='softmax')(d2)

    model = P.model.Model(inputs=i, outputs=out)

    model.compile(30, metrics=['Accuracy'], optimizer='RMSProp')
    model.summary()
    return model


def get_model_RMSProp_momentum():
    i = P.layers.Input(shape=(30))()
    d1 = P.layers.Dense(64, activation='relu')(i)
    d2 = P.layers.Dense(32, activation='relu')(d1)
    out = P.layers.Dense(2, activation='softmax')(d2)

    model = P.model.Model(inputs=i, outputs=out)
    model.compile(30, metrics=['Accuracy'], optimizer=P.optimizers.RMSProp(momentum=0.9))
    model.summary()
    return model

csvPath  = path.join(path.dirname(__file__), 'data.csv')

if __name__ == "__main__":
    dataset = Dataset(csvPath)
    model_SGD = get_model_SGD()
    model_SGD_momentum = get_model_SGD_momentum()
    model_RMSProp = get_model_RMSProp()
    model_RMSProp_momentum = get_model_RMSProp_momentum()
    model_Adam = get_model_Adam()
    model_Adam_amsgrad = get_model_Adam_amsgrad()
    model_Adagrad = get_model_Adagrad()
    x, y = dataset.features, dataset.targets
    tx, ty = x, y
    
    history_SGD = model_SGD.fit(x, y, 100, 32, verbose=False)
    history_SGD_momentum = model_SGD_momentum.fit(x, y, 100, 32, verbose=False)
    history_RMSProp = model_RMSProp.fit(x, y, 100, 32, verbose=False)
    history_RMSProp_momentum = model_RMSProp_momentum.fit(x, y, 100, 32, verbose=False)
    history_Adam = model_Adam.fit(x, y, 100, 32, verbose=False)
    history_Adam_amsgrad = model_Adam_amsgrad.fit(x, y, 100, 32, verbose=False)
    history_Adagrad = model_Adagrad.fit(x, y, 100, 32, verbose=False)
    
    print(f'model SGD: {model_SGD.evaluate((tx, ty))}')
    print(f'model RMSProp : {model_RMSProp.evaluate((tx, ty))}')
    print(f'model Adam: {model_Adam.evaluate((tx, ty))}')
    print(f'model Adagrad: {model_Adagrad.evaluate((tx, ty))}')
    
    plt.plot(history_RMSProp['accuracy'])
    plt.plot(history_RMSProp_momentum['accuracy'])
    plt.plot(history_SGD['accuracy'])
    plt.plot(history_SGD_momentum['accuracy'])
    plt.plot(history_Adam['accuracy'])
    plt.plot(history_Adam_amsgrad['accuracy'])
    plt.plot(history_Adagrad['accuracy'])

    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['RMSProp', 'RMSProp_momentum', 'SGD', 'SGD_momentum', 'Adam', 'Adam_amsgrad', 'Adagrad'], loc='lower right')
    plt.show()

    plt.plot(history_RMSProp['loss'])
    plt.plot(history_RMSProp_momentum['loss'])
    plt.plot(history_SGD['loss'])
    plt.plot(history_SGD_momentum['loss'])
    plt.plot(history_Adam['loss'])
    plt.plot(history_Adam_amsgrad['loss'])
    plt.plot(history_Adagrad['loss'])

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['RMSProp', 'RMSProp_momentum', 'SGD', 'SGD_momentum', 'Adam', 'Adam_amsgrad', 'Adagrad'], loc='upper right')
    plt.show()
