import numpy as np

class ReLu:
  def forward(self, x):
    self.x = x
    return np.maximum(0, x)

  def backward(self, grad):
    return grad * (self.x > 0)

class Sigmoid:
  def forward(self, x):
    self.out = 1 / (1 + np.exp(-x))
    return self.out

  def backward(self, grad):
    return grad * self.out * (1 - self.out)

class Tanh:
  def forward(self, x):
    self.out = np.tanh(x)
    return self.out
  
  def backward(self, grad):
    return grad * (1 - self.out ** 2)

class Softmax:
  def forward(self, x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    self.out = e / np.sum(e, axis=1, keepdims=True)
    return self.out
  
  def backward(self, grad):
    return grad