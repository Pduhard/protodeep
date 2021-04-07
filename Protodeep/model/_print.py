import numpy as np


def layer_summary(self, layer, rsize):
    param = np.sum([w.size for w in layer.get_trainable_weights()], dtype=int)
    end_str = '=' * rsize if layer is self.flatten_graph[-1] else '_' * rsize
    if isinstance(layer.output_shape, int):
        output_shape = (layer.output_shape,)
    else:
        output_shape = layer.output_shape
    if layer.input_connectors is not None:
        connected_layer = layer.input_connectors.layer.name
    else:
        connected_layer = ''
    print('{}{}{}{}\n{}'.format(
        f'{layer.name} ({layer.__class__.__name__})'.format().ljust(29),
        f'{(None, *output_shape)}'.ljust(26),
        str(param).ljust(13),
        connected_layer if not self.linear else '',
        end_str
    ))
    return param


def summary(self):
    rsize = 65 if self.linear else 94
    print('_' * rsize)
    print('{}{}{}{}\n{}'.format(
        'Layer (type)'.ljust(29),
        'Output Shape'.ljust(26),
        'Param #'.ljust(13),
        'Connected To'.ljust(29) if not self.linear else '',
        '=' * rsize
    ))
    total_param = 0
    for layer in self.flatten_graph:
        total_param += self.layer_summary(layer, rsize)
    print('Total params:', '{:,}'.format(total_param))
    print('_' * rsize)


def print_epoch_metrics(self, e, epochs):
    logs = self.logs
    log_str = 'epoch ' + str(e) + '/' + str(epochs) + ' - loss: '
    log_str += '{:.4f}'.format(logs['loss'][-1])
    for metric in self.metrics:
        log_str += ' - ' + metric.name + ': '
        log_str += '{:.4f}'.format(logs[metric.name][-1])
    if self.val_set:
        log_str += ' - val_loss: {:.4f}'.format(logs['val_loss'][-1])
        for metric in self.metrics:
            log_str += ' - val_' + metric.name + ': '
            log_str += '{:.4f}'.format(logs['val_' + metric.name][-1])
    print(log_str)
