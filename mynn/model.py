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

    def evaluate(self, X, y, batch_size=32):
        loss, acc, steps = 0, 0, 0
        for batch_X, batch_y in self._batch(X, y, batch_size):
            output = self.forward(batch_X)
            loss += self.loss.calculate(output, batch_y)
            predictions = self.layers[-1].predictions(output)
            acc += self.accuracy.calculate(predictions, batch_y)
            steps += 1
        loss = loss / steps
        acc = acc / steps
        print(f'validation loss: {loss:.3f} acc: {acc:.3f}')

    def train(self, X, y, epochs=1, print_every=1,
              batch_size=32, validation_data=None):
        self.accuracy.init(y)
        trainable_layers = [l for l in self.layers if hasattr(l, 'weights')]
        for epoch in range(epochs):
            loss, acc, steps = 0, 0, 0
            for batch_X, batch_y in self._batch(X, y, batch_size):
                output = self.forward(batch_X)
                self.backward(output, batch_y)

                loss += self.loss.calculate(output, batch_y)
                predictions = self.layers[-1].predictions(output)
                acc += self.accuracy.calculate(predictions, batch_y)

                self.optimizer.pre_update_params()
                for layer in trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                steps += 1

            loss = loss / steps
            acc = acc / steps
            if not epoch % print_every:
                print(f'epoch: {epoch+1} acc: {acc:.3f} loss: {loss:.3f} '
                      f'lr: {self.optimizer.current_learning_rate}')
                if validation_data is not None:
                    self.evaluate(*validation_data, batch_size=32)

    def _batch(self, X, y, batch_size):
        if batch_size is None:
            yield X, y
        total_steps = len(X) // batch_size
        if total_steps * batch_size < len(X):
            total_steps += 1
        for step in range(total_steps):
            batch_X = X[step*batch_size:(step+1)*batch_size]
            batch_y = y[step*batch_size:(step+1)*batch_size]
            yield batch_X, batch_y
