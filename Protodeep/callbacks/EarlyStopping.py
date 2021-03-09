from Protodeep.callbacks.CallBack import CallBack
import numpy as np


class EarlyStopping(CallBack):

    wait = 0

    def __init__(self, monitor='val_loss', min_delta=0, patience=3, verbose=0,
                 mode='auto', baseline=None, restore_best_weights=False):
        CallBack.__init__(self)
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights

    def on_fit_start(self):
        self.wait = 0
        self.best = np.inf

    def on_epoch_end(self, logs=None):
        """ for instance return True = continue, return False = stop fitting """

        if logs is None or self.monitor not in logs:
            print("Warning: EarlyStopping: \
            the given monitor have no correspondance in model metrics")
            return True

        monitor_value = logs[self.monitor][-1]
        if monitor_value < self.best:
            self.wait = 0
            self.best = monitor_value
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return False
        return True
