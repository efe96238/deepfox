import numpy as np

class Layer:
  def forward(self, x):
    raise NotImplementedError

  def backward(self, grad):
    raise NotImplementedError

  def parameters(self):
    return []

  def grads(self):
    return []

  def zero_grad(self):
    for grad in self.grads():
      grad.fill(0.0)

class Linear(Layer):
  def __init__(self, in_features, out_features):
    limit = np.sqrt(2.0 / in_features) #He initialization

    self.weights = np.random.randn(in_features, out_features) * limit
    self.bias = np.zeros((1, out_features))

    self.dweights = np.zeros_like(self.weights)
    self.dbias = np.zeros_like(self.bias)

  def forward(self, x):
    self.x = np.asarray(x)
    return self.x @ self.weights + self.bias

  def backward(self, grad):
    grad = np.asarray(grad)

    self.dweights = self.x.T @ grad
    self.dbias = np.sum(grad, axis=0, keepdims=True)

    return grad @ self.weights.T

  def parameters(self):
    return [self.weights, self.bias]

  def grads(self):
    return [self.dweights, self.dbias]