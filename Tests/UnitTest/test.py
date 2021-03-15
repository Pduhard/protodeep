import tensorflow as tf
import numpy as np


inputs = tf.convert_to_tensor(np.random.rand(32, 32), dtype=tf.float32)
test = tf.keras.activations.softmax(inputs)
ndtest = tf.make_ndarray(tf.make_tensor_proto(test))
print(ndtest)
