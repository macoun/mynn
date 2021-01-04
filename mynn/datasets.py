import numpy as np
import matplotlib.pyplot as plt


def linear(nfeatures, nsamples, weights=None, bias=None):
    X = np.random.normal(0, 1, (nsamples, nfeatures))
    weights = weights or np.random.random(nfeatures)
    bias = bias or np.random.random()
    y = np.dot(X, weights.T) + bias
    return X, y.reshape(-1, 1), (weights, bias)


# Copyright (c) 2015 Andrej Karpathy
# License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
# Source: https://cs231n.github.io/neural-networks-case-study/
def spiral_classification(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    ix = np.arange(samples*classes)
    np.random.shuffle(ix)
    for class_number in range(classes):
        ixp = ix[samples*class_number:samples*(class_number+1)]
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4, samples) + \
            np.random.randn(samples)*0.2
        X[ixp] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ixp] = class_number
    # return X, y.reshape(-1, 1)
    return X, y


if __name__ == '__main__':
    X, y = spiral_classification(samples=200, classes=2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    plt.show()
