import numpy as np
from .base import Layer

class Flatten(Layer):
  def forward(self, x):
    x = np.asarray(x)
    self.input_shape = x.shape

    batch_size = x.shape[0]

    x = x.reshape(batch_size, -1)

    return x
  
  def backward(self, grad):
    return grad.reshape(self.input_shape)
  
  def parameters(self):
    return []
  
  def get_config(self):
    return {
      "type": "Flatten"
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}()"