import numpy as np

class Network:

    def __init__(self, sizes):
        self.sizes = sizes
        self.thetas = [np.random.randn(y, x + 1) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, x):
        a = x
        for theta in self.thetas:
            a = sigmoid(theta * a)
        return a

    def backprop(self, x, y):
        h = self.feed_forward(x)



def cost()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)
