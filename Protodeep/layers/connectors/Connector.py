class Connector:

    def __init__(self, shape, layer=None):
        self.shape = shape
        self.layer = layer
        self.next_layers = []

    def __repr__(self):
        return 'shape: {} (from {})'.format(
            self.shape,
            self.layer.name if self.layer is not None else 'None'
        )
