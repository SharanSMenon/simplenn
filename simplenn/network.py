class Network():
    """
    Neural Network
    """
    def __init__(self, lr=0.01):
        self.layers = []
        self.lr = lr

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, loss_grad):
        for layer in reversed(self.layers):
            # print(loss_grad.shape)
            loss_grad = layer.backward(loss_grad, self.lr)

    def __call__(self, x):
        return self.forward(x)