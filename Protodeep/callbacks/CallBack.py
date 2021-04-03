class CallBack:

    def __init__(self):
        self.model = None

    def on_fit_start(self):
        pass

    def on_fit_end(self):
        pass

    def on_epoch_end(self, logs=None):
        pass

    def set_model(self, model):
        self.model = model
