import tensorflow as tf
import numpy as np
import time

m = tf.keras.metrics.BinaryAccuracy()
m.update_state([[0, 1], [0, 1], [0, 1], [0, 1]], [[1, 0], [0, 1], [0, 1], [0, 1]])
# m.update_state([[1], [2.00000002], [3], [4]], [[0], [2], [3], [4]])
print(m.result().numpy())

m = tf.keras.metrics.CategoricalAccuracy()
m.update_state([[0, 0.5, 1], [0, 1, 0]], [[0.8, 0.2, 0.7], [0.05, 0.95, 0]])
print(m.result().numpy())


af = np.arange(0, 20, 1).reshape(5, 2, 2)
print(af)
print(np.argmax(af, axis=0))


def a(a):
    for i in range(100000):
        a = np.square(a)


def b(b):
    for i in range(100000):
        b = b * b


ar = np.random.randn(100, 25)
br = np.random.randn(100, 25)

s = time.perf_counter()
a(ar)
print(time.perf_counter() - s)


s = time.perf_counter()
b(br)
print(time.perf_counter() - s)


epoch 0/100 - loss: 0.2742 - val_loss: 0.0376
epoch 1/100 - loss: 0.0399 - val_loss: 0.0157
epoch 2/100 - loss: 0.0129 - val_loss: 0.0083
epoch 3/100 - loss: 0.0090 - val_loss: 0.0074