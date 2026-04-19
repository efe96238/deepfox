import numpy as np
from .parameter import Parameter

class Layer:
  def forward(self, x):
    raise NotImplementedError

  def backward(self, grad):
    raise NotImplementedError

  def parameters(self):
    return []

  def zero_grad(self):
    for p in self.parameters():
      p.zero_grad()

  def get_config(self):
    return {
      "type": self.__class__.__name__
    }
  
  def __repr__(self):
    return self.__class__.__name__
  
class Sequential(Layer):
  def __init__(self, *layers):
    self.layers = list(layers)

  def forward(self, x):
    for layer in self.layers:
      x = layer.forward(x)
    return x

  def backward(self, grad):
    for layer in reversed(self.layers):
      grad = layer.backward(grad)
    return grad
  
  def parameters(self):
    params = []
    for layer in self.layers:
      params.extend(layer.parameters())
    return params
  
  def get_config(self):
    return {
      "type": "Sequential",
      "layers": [layer.get_config() for layer in self.layers]
    }
  
  def __repr__(self):
    items = [
      "\n".join("  " + line for line in repr(layer).split("\n"))
      for layer in self.layers
    ]
    joined = ",\n".join(items)
    return f"{self.__class__.__name__}(\n{joined}\n)"

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
  
class Conv1D(Layer):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    fan_in = in_channels * kernel_size
    limit = np.sqrt(2.0 / fan_in)

    self.weights = Parameter(
      np.random.randn(out_channels, in_channels, kernel_size) * limit
    )

    self.bias = Parameter(
      np.zeros((1, out_channels))
    )

  def forward(self, x):
    x = np.asarray(x)
    self.x = x

    batch_size, in_channels, input_len = x.shape

    if in_channels != self.in_channels:
      raise ValueError(f"Expected {self.in_channels} channels, got {in_channels}")

    if self.padding > 0:
      x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding)))
    else:
      x_padded = x

    self.x_padded = x_padded

    out_len = (input_len - self.kernel_size + 2 * self.padding) // self.stride + 1

    out = np.zeros((batch_size, self.out_channels, out_len))

    for b in range(batch_size):
      for oc in range(self.out_channels):
        for i in range(out_len):
          start = i * self.stride
          end = start + self.kernel_size

          window = x_padded[b, :, start:end]

          kernel = self.weights.data[oc]
          value = np.sum(window * kernel)
          value += self.bias.data[0, oc]

          out[b, oc, i] = value

    return out
  
  def backward(self, grad):
    grad = np.asarray(grad)

    batch_size, _, out_len = grad.shape

    dweights = np.zeros_like(self.weights.data)
    dbias = np.zeros_like(self.bias.data)
    dx_padded = np.zeros_like(self.x_padded)

    for b in range(batch_size):
      for oc in range(self.out_channels):
        for i in range(out_len):
          start = i * self.stride
          end = start + self.kernel_size

          window = self.x_padded[b, :, start:end]

          dweights[oc] += grad[b, oc, i] * window
          dbias[0, oc] += grad[b, oc, i]
          dx_padded[b, :, start:end] += grad[b, oc, i] * self.weights.data[oc]

    if self.padding > 0:
      dx = dx_padded[:, :, self.padding:-self.padding]
    else:
      dx = dx_padded

    self.weights.grad = dweights
    self.bias.grad = dbias

    return dx

  def parameters(self):
    return [self.weights, self.bias]

  def get_config(self):
    return {
      "type": "Conv1D",
      "in_channels": self.in_channels,
      "out_channels": self.out_channels,
      "kernel_size": self.kernel_size,
      "stride": self.stride,
      "padding": self.padding
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
  
class Conv2D(Layer):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    fan_in = in_channels * kernel_size * kernel_size
    limit = np.sqrt(2.0 / fan_in)

    self.weights = Parameter(
      np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * limit
    )

    self.bias = Parameter(
      np.zeros((1, out_channels))
    )

  def forward(self, x):
    x = np.asarray(x)
    self.x = x

    batch_size, in_channels, height, width = x.shape

    if in_channels != self.in_channels:
      raise ValueError(f"Expected {self.in_channels} channels, got {in_channels}")

    if self.padding > 0:
      x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
    else:
      x_padded = x

    self.x_padded = x_padded

    out_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
    out_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1

    out = np.zeros((batch_size, self.out_channels, out_height, out_width))

    for b in range(batch_size):
      for oc in range(self.out_channels):
        for i in range(out_height):
          for j in range(out_width):
            h_start = i * self.stride
            h_end = h_start + self.kernel_size

            w_start = j * self.stride
            w_end = w_start + self.kernel_size

            window = x_padded[b, :, h_start:h_end, w_start:w_end]

            kernel = self.weights.data[oc]
            value = np.sum(window * kernel)
            value += self.bias.data[0, oc]

            out[b, oc, i, j] = value

    return out
  
  def backward(self, grad):
    grad = np.asarray(grad)

    batch_size, _, out_height, out_width = grad.shape

    dweights = np.zeros_like(self.weights.data)
    dbias = np.zeros_like(self.bias.data)
    dx_padded = np.zeros_like(self.x_padded)

    for b in range(batch_size):
      for oc in range(self.out_channels):
        for i in range(out_height):
          for j in range(out_width):
            h_start = i * self.stride
            h_end = h_start + self.kernel_size

            w_start = j * self.stride
            w_end = w_start + self.kernel_size

            window = self.x_padded[b, :, h_start:h_end, w_start:w_end]

            dweights[oc] += grad[b, oc, i, j] * window
            dbias[0, oc] += grad[b, oc, i, j]
            dx_padded[b, :, h_start:h_end, w_start:w_end] += grad[b, oc, i, j] * self.weights.data[oc]

    if self.padding > 0:
      dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
    else:
      dx = dx_padded

    self.weights.grad = dweights
    self.bias.grad = dbias

    return dx
  
  def parameters(self):
    return [self.weights, self.bias]
  
  def get_config(self):
    return {
      "type": "Conv2D",
      "in_channels": self.in_channels,
      "out_channels": self.out_channels,
      "kernel_size": self.kernel_size,
      "stride": self.stride,
      "padding": self.padding
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"