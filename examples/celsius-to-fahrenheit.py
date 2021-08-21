import numpy as np
from simplenn import Sequential, Linear, MSELoss
import matplotlib.pyplot as plt

x_train = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=np.float32)  # Celsius
y_train = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=np.float32)  # Fahrenheit
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

criterion = MSELoss()
model = Sequential([Linear(1, 1)], lr=0.001)
EPOCHS = 20000


def plot():
    out = model(x_train)

    plt.clf()
    plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
    plt.plot(x_train, out, '--', label='Predictions', alpha=0.5)
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        out = model(x_train)
        loss = criterion.loss(y_train, out)
        error = criterion.gradient(y_train, out)
        model.backward(error)
        if epoch % 500 == 0:
            print(f"Epoch: {epoch}, Loss:{np.mean(loss)}")
    plot()
