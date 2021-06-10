import numpy as np


class Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # save inputs for later (taking derivative irt. weights)
        self.inputs = inputs
        # populate output for forward feeding
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # dy/dw for weight updates
        self.dweights = np.dot(self.inputs.T, dvalues)
        # we sum the gradients
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # the main job oj backward is to populate dinputs
        # populate dinputs for backward feeding
        self.dinputs = np.dot(dvalues, self.weights.T)
