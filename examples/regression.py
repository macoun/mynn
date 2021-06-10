import mynn

input_layer = mynn.layers.Dense(1, 2)
activation1 = mynn.activations.ReLU()

output_layer = mynn.layers.Dense(2, 1)
activation2 = mynn.activations.Linear()

loss_function = mynn.loss.MeanSquaredError()
# loss_function = mynn.loss.MeanAbsoluteError()

# optimizer = mynn.optimizers.SGD(learning_rate=0.02, decay=0.001, momentum=0.7)
# optimizer = mynn.optimizers.Adagrad(learning_rate=0.02, decay=0., epsilon=0.001)
# optimizer = mynn.optimizers.RMSProp(learning_rate=0.005, decay=1e-2, epsilon=0.001)
# optimizer = mynn.optimizers.RMSProp(learning_rate=0.005, decay=1e-2, epsilon=0.0001)
optimizer = mynn.optimizers.Adam(learning_rate=0.02, decay=1e-3, epsilon=0.000001)
accuracy = mynn.metrics.RegressionAccuracy()

model = mynn.model.Model()
model.add(input_layer)
model.add(activation1)
model.add(output_layer)
model.add(activation2)

model.set_loss(loss_function)
model.set_optimizer(optimizer)
model.set_accuracy(accuracy)

train_X, train_y, (weights, bias) = mynn.datasets.linear(1, 1000)
test_X, test_y, _ = mynn.datasets.linear(1, 200, weights, bias)

model.train(train_X, train_y,
            epochs=50,
            print_every=100,
            validation_data=(test_X, test_y))

model.evaluate(test_X, test_y)
