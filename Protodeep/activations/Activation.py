class Activation:
    """
        Activation template class
    """
    def __call__(self, inputs):
        NotImplementedError

    def derivative(self, inputs):
        NotImplementedError
