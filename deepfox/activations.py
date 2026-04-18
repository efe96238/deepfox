import numpy as np

from .layers import Layer

class ReLU(Layer):
  def forward(self, x):
    x = np.asarray(x)
    self.x = x
    return np.maximum(0, x)

  def backward(self, grad):
    grad = np.asarray(grad)
    return grad * (self.x > 0)

  def parameters(self):
    return []

class Sigmoid(Layer):
  def forward(self, x):
    x = np.asarray(x)
    self.out = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    return self.out

  def backward(self, grad):
    grad = np.asarray(grad)
    return grad * self.out * (1 - self.out)

  def parameters(self):
    return []

class Tanh(Layer):
  def forward(self, x):
    x = np.asarray(x)
    self.out = np.tanh(x)
    return self.out

  def backward(self, grad):
    grad = np.asarray(grad)
    return grad * (1 - self.out ** 2)

  def parameters(self):
    return []

class Softmax(Layer):
  def forward(self, x):
    x = np.asarray(x)

    if x.ndim == 1:
      x = x.reshape(1, -1)

    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return self.out

  def backward(self, grad):
    grad = np.asarray(grad)

    if grad.ndim == 1:
      grad = grad.reshape(1, -1)

    dot = np.sum(grad * self.out, axis=1, keepdims=True)
    return self.out * (grad - dot)

  def parameters(self):
    return []