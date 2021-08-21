import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from simplenn import MSELoss, Linear, Activation, Sequential
from simplenn.activations import Tanh

EPOCHS = 500

model = Sequential([
    Linear(2, 10, name="input"),
    Activation(Tanh(), name="tanh1"),
    Linear(10, 20, name="middle"),
    Activation(Tanh(), name="tanh2"),
    Linear(20, 1, name="output"),
    Activation(Tanh(), name="tanh3")
], lr=0.1)


def load_data():
    X, Y = make_circles(noise=0.15, factor=0.3, random_state=1, n_samples=500)
    Y = Y.reshape(-1, 1)
    Y = Y.reshape((Y.shape[0], 1))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.36, random_state=42)
    return X_train, X_test, y_train, y_test


def main():
    criterion = MSELoss()
    X_train, X_test, y_train, y_test = load_data()
    for epoch in range(1, EPOCHS + 1):
        out = model(X_train)
        loss = criterion.loss(y_train, out)
        acc = (np.where(out > 0.5, 1, 0) == y_train).mean()
        error = criterion.gradient(y_train, out)
        model.backward(error)
        if epoch % 50 == 0:
            print(f"Epoch: {epoch}, Loss:{np.mean(loss)}, Acc: {acc}")


if __name__ == "__main__":
    main()
