import numpy as np

class ReLU:
  def forward(self, x):
    self.x = np.asarray(x)
    return np.maximum(0.0, self.x)

  def backward(self, grad):
    grad = np.asarray(grad)
    return grad * (self.x > 0)

class Sigmoid:
  def forward(self, x):
    x = np.asarray(x)
    x = np.clip(x, -500, 500)
    self.out = 1.0 / (1.0 + np.exp(-x))
    return self.out

  def backward(self, grad):
    grad = np.asarray(grad)
    return grad * self.out * (1.0 - self.out)

class Tanh:
  def forward(self, x):
    x = np.asarray(x)
    self.out = np.tanh(x)
    return self.out

  def backward(self, grad):
    grad = np.asarray(grad)
    return grad * (1.0 - self.out ** 2)

class Softmax:
  def forward(self, x):
    x = np.asarray(x)

    if x.ndim == 1:
      x = x.reshape(1, -1)

    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return self.out

  def backward(self, grad):
    return np.asarray(grad)