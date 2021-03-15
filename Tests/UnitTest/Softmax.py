import unittest
import numpy as np
import tensorflow as tf


from Protodeep.activations.Softmax import Softmax


class TestSoftmax(unittest.TestCase):

    def test_call2d(self):
        inputs = np.random.rand(32, 32)
        inputstf = tf.convert_to_tensor(inputs)
        # inputstf = tf.convert_to_tensor(inputs, dtype=tf.float64)
        sftmaxtf = tf.keras.activations.softmax(inputstf)
        tfnd = tf.make_ndarray(tf.make_tensor_proto(sftmaxtf))
        # test = tf.keras.activations.softmax()(inputs)
        # print(type(test))
        self.assertEqual(np.round(Softmax()(inputs), 8).tolist(), np.round(tfnd, 8).tolist())
        # self.assertEqual('foo'.upper(), 'FOO')

    # def test_dr(self):  fucck
        # with tf.GradientTape(persistent=True) as tape:
            # Make prediction
            # pred_y = tf.keras.activations.softmax(inputstf) * (1 - tf.keras.activations.softmax(inputstf))
            # Calculate loss
            # poly_loss = loss(real_y, pred_y)

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()