from Protodeep.callbacks.CallBack import CallBack
import numpy as np
import copy


class EarlyStopping(CallBack):

    def __init__(self, monitor='val_loss', min_delta=0, patience=3,
                 baseline=None, restore_best_weights=False):
        CallBack.__init__(self)
        self.wait = 0
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.over_baseline = not baseline
        self.baseline = baseline or np.inf
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
        if monitor_value < self.best - self.min_delta:
            self.wait = 0
            self.best = monitor_value
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(self.model.get_weights())
        elif self.over_baseline or monitor_value < self.baseline:
            self.wait += 1
            self.over_baseline = True
            if self.wait >= self.patience:
                if self.restore_best_weights:
                    self.model.set_weights(self.best_weights)
                return False
        return True
