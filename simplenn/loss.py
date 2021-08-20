import numpy as np

class Loss():
    def __init__(self): pass

    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y_true, y_pred):
        return NotImplementedError()


class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2));

    def gradient(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size;


class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)