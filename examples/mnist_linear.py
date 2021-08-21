import numpy as np
import tensorflow as tf  # For data only
from simplenn import Network, Linear, CrossEntropyLoss, Activation
from simplenn.activations import LeakyReLU, Softmax
from simplenn.data_utils import to_categorical, batch_generator, accuracy_score
from tqdm import tqdm

EPOCHS = 5


class NeuralNetwork(Network):
    def __init__(self, input_dim, output_dim, lr=0.01):
        super().__init__(lr)
        self.layers = [
            Linear(input_dim, 256, name="input"),
            Activation(LeakyReLU(), name="relu1"),
            Linear(256, 128, name="middle"),
            Activation(LeakyReLU(), name="relu2"),
            Linear(128, output_dim, name="output"),
            Activation(Softmax(), name="softmax")
        ]


def load_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    y_train, y_test = to_categorical(y_train.astype("int")), to_categorical(y_test.astype("int"))
    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train, X_test = X_train.reshape(-1, 28 * 28), X_test.reshape(-1, 28 * 28)
    return X_train, y_train, X_test, y_test


def main():
    model = NeuralNetwork(784, 10, lr=1e-3)
    criterion = CrossEntropyLoss()
    X_train, y_train, X_test, y_test = load_data()
    for epoch in range(EPOCHS):
        loss = []
        acc = []
        for x_batch, y_batch in batch_generator(X_train, y_train, batch_size=256):
            out = model.forward(x_batch)
            loss.append(np.mean(criterion.loss(y_batch, out)))
            acc.append(accuracy_score(np.argmax(y_batch, axis=1), np.argmax(out, axis=1)) * 100)
            error = criterion.gradient(y_batch, out)
            model.backward(error)
        if epoch % 1 == 0:
            print("Epoch: {}, Loss: {}, Acc: {}".format(epoch, np.mean(loss), np.mean(acc)))


if __name__ == "__main__":
    main()
