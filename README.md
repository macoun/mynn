This is a from-scratch implementation of neural networks depending only on numpy. 
It is for educational purposes only.

## Concepts
### Neuron

Linear equation 

	y = w1*x1 + ... + wn*xn + b

Input is a vector
	
    X = [x1 ... xn]

and the output is a scalar

	y

### Layer

List of linear equations in the same dimension _n_.

	y1 = w11*x1 + w12*x2 + ... + w1n*xn + b1
    ...
	ym = wm1*x1 + wm2*x2 + ... + wmn*xn + bm
 
    
Therefore the output of _m_ linear equations for one sample, i.e. same input `[x1 ... x2]`, is

	y = [y1 ... ym]


We want to calculate results for multiple samples at once. In other words, we want to feed a list of samples and receive a list of results from a layer. 

Input for _k_ samples

	X = [[x11 ... x1n] ... [xk1 ... xkn]]

Output for _k_ samples is a list of results for each _m_ equation

	y = [[y11 ... y1m] ... [yk1 ... ykm]]


A layer stores its weights and biases

	w = [[w11 .. wm1] [w12 .. wm2] ... [w1n .. wmn]]
    b = [b1 ... bm]

Shapes of input _X_, weights _w_, biases _b_, and output _y_ is _(k, n), (n, m), (k, ), (k, m)_

The calculation of the puts is then done with a simple dot product of inputs and weights
	
    y = X*w + b
    
We get a dimension of _(k, n) * (n, m) + (k,) -> (k, m)_

Functional representation of a layer

	layer(X) = X*w + b


### Activation Functions

An activation function _a_ is applied to the output of each neuron in a layer

	a(y)

We will use the same activation function for all neurons in a layer.

Input is the output of a layer of shape _(k, m)_, where _k_ is the number of samples and _m_ the number of neurons.

Output is usually the same shape _(k, m)_, which will be used as input for the next layer or for the loss function.

#### Rectified Linear Activation Function (ReLU)

The ReLU activation function is extremely close to being a linear activation function while remaining nonlinear, due to that bend after 0.

	a(y) = max(0, y)

### Feed forward

Putting neurons, layers, activation functions together we can get an output (_ŷ_) from our network

	ŷ = a(layer(X))

### Loss Functions

Loss function (aka cost function) is the algorithm that quantifies how wrong a network is.

Loss is the measure of this metric. 

The input for a loss function  _C_ is the output of the activation function of the last layer (_ŷ_). 

	C(ŷ, y) = C(a(layer(X)), y)

The output of a lost function is the loss for each sample.

	L = [l1 ... lk]

The shape of the output is _(k,)_

    C(ŷ, y) - > L


The overall loss of a network is the mean of the elements in _L_ plus some regulation errors _L1_ and _L2_ (if applied).

#### Mean Squared Error Loss

You square the difference between the predicted (_ŷ_) and true values (_y_) (as the model can have multiple regression outputs) and average those squared values.

	C(ŷ, y) = sum((y - ŷ)**2)/k

where _k_ is the number of samples

### Optimization

The only values we can change to get a better loss (optimally zero) in a network is to change the weights and biases in the layers.

#### Gradient Descent

We need the gradient of the loss function _C_ with respect to the weights to get the direction and magnitude to move the weights towards.

	w = w - r*∇(dC/dw)

Where _r_ is a value between (0,1) and is called the learning rate. 


#### Chain Rule

Recall, to calculate loss (_l_ stands for _layer_)
	
    C(ŷ, y) = C(a(l(X)), y)

According to chain rule, the gradient of C with respect to w can be written as

	dC/dw = (dl/dw)*(da/dl)*(dC/da)

#### Partial Derivatives

The derivative of Mean Squared Error function is (shape _(k, m)_)

	dC_da = -2 * (y - ŷ) / k

The derivative of ReLU activation function is (shape _(k, m)_)

	da_dl = dC_da
    da_dl[layer(X) <= 0] = 0

The derivative of the layer is

	dl_dw = X.T * da_dl

Shape of `dl_dw` is _(n, k) * (k, m) -> (n, m)_

This means update _m_ weights for each input in _n_.

To update the biases we will aggregate all samples and get a list of sum of weights for each input.

	dl_db = sum(da_dl) through axis=0
    
Shape of `dl_db` is _(n, m)_.




#### Vanilla SGD (Stochastic Gradient Descent) Optimizer

We apply the partial derivatives using gradient descent and the chain rule.
	
	w = w - r*dl_dw
	b = b - r*dl_db


## Implementation
### Metrics
### Feed Forward
### Feed Backward
### Model
### Predictiion
### Evaluation























