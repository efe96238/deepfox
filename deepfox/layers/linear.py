import numpy as np
from ..parameter import Parameter
from .base import Layer

class Linear(Layer):
  def __init__(self, in_features, out_features):
    limit = np.sqrt(2.0 / in_features)

    self.weights = Parameter(
      np.random.randn(in_features, out_features) * limit
    )

    self.bias = Parameter(
      np.zeros((1, out_features))
    )

  def forward(self, x):
    self.x = np.asarray(x)
    return self.x @ self.weights.data + self.bias.data

  def backward(self, grad):
    grad = np.asarray(grad)

    self.weights.grad = self.x.T @ grad
    self.bias.grad = np.sum(grad, axis=0, keepdims=True)

    return grad @ self.weights.data.T

  def parameters(self):
    return [self.weights, self.bias]
  
  def get_config(self):
    in_features, out_features = self.weights.data.shape

    return {
      "type": "Linear",
      "in_features": int(in_features),
      "out_features": int(out_features)
    }
  
  def __repr__(self):
    in_features, out_features = self.weights.data.shape
    return f"{self.__class__.__name__}(in_features={in_features}, out_features={out_features})"