import numpy as np


class ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs


class Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for i, (s_output, s_dvalues) in enumerate(zip(self.output, dvalues)):
            s_output = s_output.reshape(-1, 1)
            jacobian_mtx = np.diagflat(s_output) - np.dot(s_output, s_output.T)
            self.dinputs[i] = np.dot(jacobian_mtx, s_dvalues)

    def predictions(self, outputs):
        predictions = np.argmax(outputs, axis=1)
        return predictions


class Linear:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues

    def predictions(self, outputs):
        return outputs


class Sigmoid:

    def forward(self, inputs):
        # self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1
