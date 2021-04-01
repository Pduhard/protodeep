import tensorflow as tf
import numpy as np

i = tf.keras.layers.Input((3, 3, 3))
conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), activation='relu')(i)
f = tf.keras.layers.Flatten()(conv)
out = tf.keras.layers.Dense(1)(f)

model = tf.keras.Model(inputs=i, outputs=out)

model.compile()
model.summary()

x, y = np.random.randn(100, 45), np.random.randn(100, 1)

model.fit(x, y)