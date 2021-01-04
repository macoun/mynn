class Model:

    def __init__(self, layers=[]):
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        self.loss = loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_accuracy(self, accuracy):
        self.accuracy = accuracy

    def forward(self, X):
        output = X
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def backward(self, output, y):
        self.loss.backward(output, y)
        dinputs = self.loss.dinputs
        for layer in reversed(self.layers):
            layer.backward(dinputs)
            dinputs = layer.dinputs

    def train(self, X, y, epochs=1, print_every=1):
        self.accuracy.init(y)
        trainable_layers = [l for l in self.layers if hasattr(l, 'weights')]
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(output, y)

            loss = self.loss.calculate(output, y)
            predictions = self.layers[-1].predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            if not epoch % print_every:
                print(f'epoch: {epoch+1} acc: {accuracy} loss: {loss} '
                      f'lr: {self.optimizer.current_learning_rate}')
                pass

            self.optimizer.pre_update_params()
            for layer in trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()
