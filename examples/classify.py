import mynn

input_layer = mynn.layers.Dense(2, 64)
activation1 = mynn.activations.ReLU()

hidden_layer = mynn.layers.Dense(64, 64)
activation2 = mynn.activations.ReLU()

output_layer = mynn.layers.Dense(64, 3)
activation3 = mynn.activations.Softmax()


loss_function = mynn.loss.CategoricalCrossentropy()

# optimizer = mynn.optimizers.SGD(learning_rate=0.02, decay=0.001, momentum=0.7)
optimizer = mynn.optimizers.Adagrad(learning_rate=0.0273, decay=0., epsilon=0.001)
# optimizer = mynn.optimizers.RMSProp(learning_rate=0.05, decay=1e-2, epsilon=0.001)
# optimizer = mynn.optimizers.RMSProp(learning_rate=0.05, decay=1e-2, epsilon=0.0001)
# optimizer = mynn.optimizers.Adam(
    # learning_rate=0.05, decay=5e-2, epsilon=0.0001)
accuracy = mynn.metrics.CategoricalAccuracy()

model = mynn.model.Model()
model.add(input_layer)
model.add(activation1)
model.add(hidden_layer)
model.add(activation2)
model.add(output_layer)
model.add(activation3)

model.set_loss(loss_function)
model.set_optimizer(optimizer)
model.set_accuracy(accuracy)

X, y = mynn.datasets.spiral_classification(1000, 3)
validation_split = 0.8
split_idx = int(len(X)*validation_split)
train_X, train_y = X[:split_idx], y[:split_idx]
test_X, test_y = X[split_idx:], y[split_idx:]
model.train(train_X, train_y, epochs=300, print_every=10,
            validation_data=(test_X, test_y), batch_size=32)
model.evaluate(test_X, test_y)
