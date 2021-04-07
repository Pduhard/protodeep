import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(func):
        return func

from Protodeep.utils.parse import parse_activation
from Protodeep.utils.parse import parse_initializer, parse_regularizer
from Protodeep.utils.debug import class_timer
from Protodeep.layers.Layer import Layer


@class_timer
class LSTM(Layer):
    """
        LSTM layer (no LSTM cell all is computed here)

        variables explanation:

            wf = weights for forget gate
            wi = weights for input gate
            wo = weights for output gate
            wc = weights for candidate gate

            hwf = hidden weights for forget gate
            hwi = hidden weights for input gate
            hwo = hidden weights for output gate
            hwc = hidden weights for candidate gate

            bf = biases for forget gate
            bi = biases for input gate
            bo = biases for output gate
            bc = biases for candidate gate


            wf_g = gradient of weights for forget gate
            wi_g = gradient of weights for input gate
            wo_g = gradient of weights for output gate
            wc_g = gradient of weights for candidate gate

            hwf_g = gradient of hidden weights for forget gate
            hwi_g = gradient of hidden weights for input gate
            hwo_g = gradient of hidden weights for output gate
            hwc_g = gradient of hidden weights for candidate gate

            bf_g = gradient of biases for forget gate
            bi_g = gradient of biases for input gate
            bo_g = gradient of biases for output gate
            bc_g = gradient of biases for candidate gate

            hs = hidden state
            cs = cell state

            zf = preactivation for forget gate
            zi = preactivation for input gate
            zo = preactivation for output gate
            zc = preactivation for candidate gate

            af = activation for forget gate
            ai = activation for input gate
            ao = activation for output gate
            ac = activation for candidate gate

            out_val = full output
            a_val = formated output (used in network)

    """
    total_instance = 0

    def __init__(self, units, activation='tanh',
                 recurrent_activation='sigmoid', use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='glorot_uniform',
                 bias_initializer='zeros', unit_forget_bias=True,
                 kernel_regularizer=None, recurrent_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None,
                 return_sequences=False, name='lstm'):
        super().__init__(trainable=True, name=name)

        self.i_val = None
        self.dloss = None

        self.units = units
        self.activation = parse_activation(activation)
        self.recurrent_activation = parse_activation(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = parse_initializer(kernel_initializer)
        self.recurrent_initializer = parse_initializer(recurrent_initializer)
        self.unit_forget_bias = unit_forget_bias
        if unit_forget_bias:
            self.bias_initializer = parse_initializer('zeros')
        else:
            self.bias_initializer = parse_initializer(bias_initializer)
        self.kernel_regularizer = parse_regularizer(kernel_regularizer)
        self.recurrent_regularizer = parse_regularizer(recurrent_regularizer)
        self.bias_regularizer = parse_regularizer(bias_regularizer)
        self.activity_regularizer = parse_regularizer(activity_regularizer)
        self.return_sequences = return_sequences
        self.output_shape = (
            self.units
        )

    def __call__(self, connectors):
        # if isinstance(connectors.shape, tuple):
        #     connectors.shape = connectors.shape[-1]
        print(connectors.shape)
        timesteps, feature = connectors.shape

        weight_shape = (feature, self.units)
        hidden_weight_shape = (self.units, self.units)
        # gate_shape = (timesteps, )

        # print(weight_shape)
        # quit()
        self.wf = self.recurrent_initializer(weight_shape)
        self.wi = self.recurrent_initializer(weight_shape)
        self.wo = self.recurrent_initializer(weight_shape)
        self.wc = self.kernel_initializer(weight_shape)

        self.hwf = self.recurrent_initializer(hidden_weight_shape)
        self.hwi = self.recurrent_initializer(hidden_weight_shape)
        self.hwo = self.recurrent_initializer(hidden_weight_shape)
        self.hwc = self.kernel_initializer(hidden_weight_shape)

        self.wf_g = np.zeros(weight_shape)
        self.wi_g = np.zeros(weight_shape)
        self.wo_g = np.zeros(weight_shape)
        self.wc_g = np.zeros(weight_shape)

        self.hwf_g = np.zeros(hidden_weight_shape)
        self.hwi_g = np.zeros(hidden_weight_shape)
        self.hwo_h = np.zeros(hidden_weight_shape)
        self.hwc_g = np.zeros(hidden_weight_shape)

        if self.use_bias:
            self.bf = self.bias_initializer(self.units)
            self.bi = self.bias_initializer(self.units)
            self.bo = self.bias_initializer(self.units)
            self.bc = self.bias_initializer(self.units)
            if self.unit_forget_bias:
                self.bf += 1

            self.bf_g = np.zeros(self.units)
            self.bi_g = np.zeros(self.units)
            self.bo_g = np.zeros(self.units)
            self.bc_g = np.zeros(self.units)

        self.input_connectors = connectors

        self.output_shape = (
            timesteps,
            self.units
        ) if self.return_sequences else (self.units)

        self.output_connectors = self.new_output_connector(
            self.output_shape
        )
        return self.output_connectors

    def reset_gradients(self):
        self.dloss = np.zeros(shape=(self.i_val.shape))
        self.wf_g.fill(0)
        self.wi_g.fill(0)
        self.wo_g.fill(0)
        self.wc_g.fill(0)

        self.hwf_g.fill(0)
        self.hwi_g.fill(0)
        self.hwo_h.fill(0)
        self.hwc_g.fill(0)

        if self.use_bias:
            self.bf_g.fill(0)
            self.bi_g.fill(0)
            self.bo_g.fill(0)
            self.bc_g.fill(0)

    def _init_cache(self, batch_size, timestep):
        self.af = np.zeros(shape=(batch_size, timestep + 1, self.units))
        self.ai = np.zeros(shape=(batch_size, timestep + 1, self.units))
        self.ao = np.zeros(shape=(batch_size, timestep + 1, self.units))
        self.ac = np.zeros(shape=(batch_size, timestep + 1, self.units))

        self.zf = np.zeros(shape=(batch_size, timestep + 1, self.units))
        self.zi = np.zeros(shape=(batch_size, timestep + 1, self.units))
        self.zo = np.zeros(shape=(batch_size, timestep + 1, self.units))
        self.zc = np.zeros(shape=(batch_size, timestep + 1, self.units))

        self.hs = np.zeros(shape=(batch_size, timestep + 1, self.units))
        self.cs = np.zeros(shape=(batch_size, timestep + 1, self.units))
        self.out_val = np.zeros(shape=(batch_size, timestep, self.units))

    def regularize(self):

        if self.kernel_regularizer:
            self.wc_g += self.kernel_regularizer.derivative(self.wc)
            self.hwc_g += self.kernel_regularizer.derivative(self.hwc)

        if self.recurrent_regularizer:
            self.wf_g += self.recurrent_regularizer.derivative(self.wf)
            self.wi_g += self.recurrent_regularizer.derivative(self.wi)
            self.wo_g += self.recurrent_regularizer.derivative(self.wo)
            self.hwf_g += self.recurrent_regularizer.derivative(self.hwf)
            self.hwi_g += self.recurrent_regularizer.derivative(self.hwi)
            self.hwo_g += self.recurrent_regularizer.derivative(self.hwo)

        if self.use_bias and self.bias_regularizer:
            self.bf_g += self.bias_regularizer.derivative(self.bf)
            self.bi_g += self.bias_regularizer.derivative(self.bi)
            self.bo_g += self.bias_regularizer.derivative(self.bo)
            self.bc_g += self.bias_regularizer.derivative(self.bc)

    def cell_forward_pass(self, b, i, x):
        j = (b, i - 1)
        i = (b, i)
        if self.use_bias:
            self.zf[i] += x @ self.wf + self.hs[j] @ self.hwf + self.bf
            self.zi[i] += x @ self.wi + self.hs[j] @ self.hwi + self.bi
            self.zo[i] += x @ self.wo + self.hs[j] @ self.hwo + self.bo
            self.zc[i] += x @ self.wc + self.hs[j] @ self.hwc + self.bc
        else:
            self.zf[i] += x @ self.wf + self.hs[j] @ self.hwf
            self.zi[i] += x @ self.wi + self.hs[j] @ self.hwi
            self.zo[i] += x @ self.wo + self.hs[j] @ self.hwo
            self.zc[i] += x @ self.wc + self.hs[j] @ self.hwc

        self.af[i] += self.recurrent_activation(self.zf[i])
        self.ai[i] += self.recurrent_activation(self.zi[i])
        self.ao[i] += self.recurrent_activation(self.zo[i])
        self.ac[i] += self.activation(self.zc[i])

        self.cs[i] += self.cs[j] * self.af[i] + self.ai[i] * self.ac[i]
        self.hs[i] += self.activation(self.cs[i]) * self.ao[i]
        self.out_val[j] += self.activation(self.cs[i]) * self.ao[i]

    def cell_backward_pass(self, dly, dlcsnext, b, i, batch_size, inpt):

        rad = self.recurrent_activation.derivative
        ad = self.activation.derivative

        index = i
        p = (b, i - 1)
        i = (b, i)
        dlcs = dly * self.ao[i] * ad(self.cs[i])

        # derivative of loss with respect to gates output
        dlzf = self.cs[p] * dlcsnext + dlcs * self.cs[p] * rad(self.zf[i])
        dlzi = self.ac[i] * dlcsnext + dlcs * self.ac[i] * rad(self.zi[i])
        dlzc = self.ai[i] * dlcsnext + dlcs * self.ai[i] * ad(self.zc[i])
        dlzo = dly * self.activation(self.cs[i]) * rad(self.zo[i])

        # derivative of loss with respect to weights
        self.wf_g += np.outer(dlzf, self.i_val[p]).T / batch_size
        self.wi_g += np.outer(dlzi, self.i_val[p]).T / batch_size
        self.wc_g += np.outer(dlzc, self.i_val[p]).T / batch_size
        self.wo_g += np.outer(dlzo, self.i_val[p]).T / batch_size

        # derivative of loss with respect to hidden weights
        self.hwf_g += np.outer(dlzf, self.hs[p]).T / batch_size
        self.hwi_g += np.outer(dlzi, self.hs[p]).T / batch_size
        self.hwc_g += np.outer(dlzc, self.hs[p]).T / batch_size
        self.hwo_h += np.outer(dlzo, self.hs[p]).T / batch_size

        # derivative of loss with respect to biases
        if self.use_bias:
            self.bf_g += dlzf / batch_size
            self.bi_g += dlzi / batch_size
            self.bc_g += dlzc / batch_size
            self.bo_g += dlzo / batch_size

        # derivative of loss with respect to input of cell
        self.dloss[p] += (self.wf @ dlzf + self.wi @ dlzi +
                          self.wc @ dlzc + self.wo @ dlzo).T

        # derivative of loss with respect to hidden input of previous cell
        dly = (self.hwf @ dlzf + self.hwi @ dlzi +
               self.hwc @ dlzc + self.hwo @ dlzo).T
        if self.return_sequences and index > 1:
            dly += inpt[index - 2]
        dlcsnext = dlcsnext * self.af[i] + dlcs * self.af[i]
        return dly, dlcsnext

    def forward_pass(self, inputs):
        self.i_val = inputs
        batch_size, timestep, _ = inputs.shape
        self._init_cache(batch_size, timestep)
        for b, inpt in enumerate(inputs):
            for i, x in enumerate(inpt, start=1):
                self.cell_forward_pass(b, i, x)
        if self.return_sequences:
            self.a_val = self.out_val
        else:
            self.a_val = self.out_val[:, -1, :]
        return self.a_val

    def backward_pass(self, inputs):
        """
            inputs: derivative of loss with respect to output of this layer

            outputs:
                list of gradients (same order as get_trainable_weights),
                and derivative of loss with respect to input of this layer
        """
        batch_size, timesteps, _ = self.i_val.shape
        self.reset_gradients()

        if self.activity_regularizer:
            inputs = inputs + self.activity_regularizer(inputs)
        for b, inpt in enumerate(inputs):
            dly = inpt[-1, :] if self.return_sequences else inpt
            dlcsnext = 0
            for i in range(timesteps, 0, -1):
                # derivative of loss with respect to cell state
                dly, dlcsnext = self.cell_backward_pass(
                    dly, dlcsnext, b, i, batch_size, inpt)

        self.regularize()

        return self.get_gradients(), self.dloss

    def get_trainable_weights(self):
        return [
            self.wf,
            self.wi,
            self.wo,
            self.wc,
            self.hwf,
            self.hwi,
            self.hwo,
            self.hwc,
            self.bf,
            self.bi,
            self.bo,
            self.bc
        ] if self.use_bias else [
            self.wf,
            self.wi,
            self.wo,
            self.wc,
            self.hwf,
            self.hwi,
            self.hwo,
            self.hwc
        ]

    def get_gradients(self):
        return [
            self.wf_g,
            self.wi_g,
            self.wo_g,
            self.wc_g,
            self.hwf_g,
            self.hwi_g,
            self.hwo_h,
            self.hwc_g,
            self.bf_g,
            self.bi_g,
            self.bo_g,
            self.bc_g
        ] if self.use_bias else [
            self.wf_g,
            self.wi_g,
            self.wo_g,
            self.wc_g,
            self.hwf_g,
            self.hwi_g,
            self.hwo_h,
            self.hwc_g
        ]

    def set_weights(self, weights):
        if len(weights) != len(self.get_trainable_weights()):
            print('invalid weights list lstm')
        self.wf = weights[0]
        self.wi = weights[1]
        self.wo = weights[2]
        self.wc = weights[3]
        self.hwf = weights[4]
        self.hwi = weights[5]
        self.hwo = weights[6]
        self.hwc = weights[7]
        if self.use_bias:
            self.bf = weights[8]
            self.bi = weights[9]
            self.bo = weights[10]
            self.bc = weights[11]
