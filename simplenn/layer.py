import numpy as np
# from scipy import signal
import math


class Activation():
    """
    Wrapper layer for activation functions.
    """
    def __init__(self, activation, name="activation"):
        self.activation = activation.activation
        self.activation_prime = activation.gradient
        self.input = None
        self.name = name
        self.output = None

    def forward(self, x):
        self.input = x
        self.output = self.activation(x)
        return self.output

    def backward(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

    def __call__(self, x):
        return self.forward(x)


class Linear():
    """
    Linear Layer

    **Example**:

    .. code-block:: python

        >>> a = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        >>> l = Linear(1, 1)
        >>> print(l(a))
        array([[-0.29296088],
       [-0.89514827],
       [-1.49733566],
       [-2.09952306],
       [-2.70171045]])

    """
    def __init__(self, n_in, n_out, name="layer"):
        limit = 1 / np.sqrt(n_in)
        self.W = np.random.uniform(-limit, limit, (n_in, n_out))
        self.b = np.random.rand(1, n_out)  # Biases
        self.name = name
        self.input = None
        self.output = None
        self.n_in = n_in
        self.n_out = n_out

    def forward(self, x):
        self.input = x
        self.output = np.dot(self.input, self.W) + self.b
        return self.output

    def __call__(self, x):
        return self.forward(x)

    def backward(self, error, lr=0.01):
        input_error = np.dot(error, self.W.T)
        delta = np.dot(self.input.T, error)

        self.W -= lr * delta
        self.b -= lr * np.mean(error)
        return input_error

    def __repr__(self) -> str:
        return f"({self.name}): Linear({self.n_in} -> {self.n_out})"


class Conv2d():
    """
    Conv2D layer
    Input shape: (channels, height, width)
    """
    # Input Shape (depth, height, width)
    # MNIST: 1, 28, 28
    def __init__(self, input_shape, out_channels, kernel_size, stride=1, name="layer", padding="valid", bias=False):
        in_channels, height, width = input_shape
        self.input_shape = input_shape
        self.in_channels = in_channels  # Number of channels
        self.out_channels = out_channels  # Number of out channels
        self.kernel_size = kernel_size  # Kernel size
        self.stride = stride  # Stride
        self.name = name
        self.padding = padding
        self.input = None
        self.output = None
        # Output Shape
        self.output_shape = self.out_shape()
        # Weights and biases
        limit = 1 / np.sqrt(kernel_size * kernel_size)
        self.W = np.random.uniform(-limit, limit, size=(out_channels,
                                                        in_channels, kernel_size, kernel_size))
        self.b = np.zeros((self.out_channels, 1))
        self.bias = bias

    # returns output for a given input

    def out_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = self.determine_padding(
            self.kernel_size, output_shape=self.padding)
        output_height = (height + np.sum(pad_h) -
                         self.kernel_size) / self.stride + 1
        output_width = (width + np.sum(pad_w) -
                        self.kernel_size) / self.stride + 1
        return self.out_channels, int(output_height), int(output_width)

    def forward(self, X):
        self.input = X
        batch_size, channels, height, width = X.shape

        self.X_col = self.image_to_column(
            X, self.kernel_size, self.stride, output_shape=self.padding)
        self.W_col = self.W.reshape(self.out_channels, -1)
        self.output = np.dot(self.W_col, self.X_col) + self.b
        self.output = self.output.reshape(self.output_shape + (batch_size,))
        return self.output.transpose(3, 0, 1, 2)

    def __call__(self, x):
        return self.forward(x)

    def backward(self, output_error, lr=0.01):
        output_error = output_error.transpose(
            1, 2, 3, 0).reshape(self.out_channels, -1)

        grad_w = output_error.dot(self.X_col.T).reshape(self.W.shape)
        grad_b = np.sum(output_error, axis=1, keepdims=True)

        self.W -= lr * grad_w
        if self.bias:
            self.b -= lr * grad_b

        output_error = self.W_col.T.dot(output_error)
        output_error = self.column_to_image(output_error,
                                            self.input.shape,
                                            self.kernel_size,
                                            stride=self.stride,
                                            output_shape=self.padding)
        return output_error

    def column_to_image(self, cols, images_shape, kernel_size, stride=1, output_shape='valid'):
        batch_size, channels, height, width = images_shape
        pad_h, pad_w = self.determine_padding(kernel_size, output_shape)

        height_padded = height + np.sum(pad_h)
        width_padded = width + np.sum(pad_w)
        images_padded = np.zeros(
            (batch_size, channels, height_padded, width_padded))

        k, i, j = self.get_im2col_indices(
            images_shape, kernel_size, (pad_h, pad_w), stride)

        cols = cols.reshape(channels * kernel_size *
                            kernel_size, -1, batch_size)
        cols = cols.transpose(2, 0, 1)
        np.add.at(images_padded, (slice(None), k, i, j), cols)
        return images_padded[:, :, pad_h[0]:height + pad_h[0], pad_w[0]:width + pad_w[0]]

    def image_to_column(self, images, kernel_size, stride, output_shape='valid'):
        filter_height, filter_width = kernel_size, kernel_size

        pad_h, pad_w = self.determine_padding(kernel_size, output_shape)

        # Add padding to the image
        images_padded = np.pad(
            images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')

        # Calculate the indices where the dot products are to be applied between weights
        # and the image
        k, i, j = self.get_im2col_indices(
            images.shape, kernel_size, (pad_h, pad_w), stride)

        # Get content from image at those indices
        cols = images_padded[:, k, i, j]
        channels = images.shape[1]
        # Reshape content into column shape
        cols = cols.transpose(1, 2, 0).reshape(
            filter_height * filter_width * channels, -1)
        return cols

    @staticmethod
    def get_im2col_indices(images_shape, kernel_size, padding, stride=1):
        batch_size, channels, height, width = images_shape
        filter_height, filter_width = kernel_size, kernel_size
        pad_w, pad_h = padding
        out_h = int((height + np.sum(pad_h) - filter_height) / stride + 1)
        out_w = int((width + np.sum(pad_w) - filter_width) / stride + 1)

        i0 = np.repeat(np.arange(filter_height), filter_width)
        i0 = np.tile(i0, channels)
        i1 = stride * np.repeat(np.arange(out_h), out_w)
        j0 = np.tile(np.arange(filter_width), filter_height * channels)
        j1 = stride * np.tile(np.arange(out_w), out_h)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(channels), filter_height *
                      filter_width).reshape(-1, 1)

        return (k, i, j)

    @staticmethod
    def determine_padding(kernel_size, output_shape="same"):
        if output_shape == "valid":
            return (0, 0), (0, 0)
        # Pad so that the output shape is the same as input shape (given that stride=1)
        elif output_shape == "same":
            filter_height, filter_width = kernel_size, kernel_size

            # Derived from:
            # output_height = (height + pad_h - filter_height) / stride + 1
            # In this case output_height = height and stride = 1. This gives the
            # expression for the padding below.
            pad_h1 = int(math.floor((filter_height - 1) / 2))
            pad_h2 = int(math.ceil((filter_height - 1) / 2))
            pad_w1 = int(math.floor((filter_width - 1) / 2))
            pad_w2 = int(math.ceil((filter_width - 1) / 2))

            return (pad_h1, pad_h2), (pad_w1, pad_w2)


class Flatten():
    """
    Flatten layer.
    """
    def __init__(self, name="flatten"):
        self.input = None
        self.name = name
        self.output = None

    def forward(self, x):
        self.input = x
        self.output = x.reshape(x.shape[0], -1)
        return self.output

    def __call__(self, x):
        return self.forward(x)

    def backward(self, outgrad, lr=0.01):
        return outgrad.reshape(self.input.shape)
