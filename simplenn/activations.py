import numpy as np


class Softmax():
    def __init__(self): pass

    def activation(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)

    def __call__(self, x):
        return self.activation(x)


class LeakyReLU():
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return self.activation(x)

    def activation(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)


class Tanh():
    def __init__(self): pass

    def __call__(self, x):
        return self.activation(x)

    def activation(self, x):
        return np.tanh(x)

    def gradient(self, x):
        return 1 - np.tanh(x) ** 2


class ReLU():
    def __init__(self): pass

    def __call__(self, x):
        return self.activation(x)

    def activation(self, x):
        return np.maximum(0, x)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)


class Sigmoid():
    def __init__(self): pass

    def __call__(self, x):
        return self.activation(x)

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        x = 1 / (1 + np.exp(-x))
        return x * (1 - x)
