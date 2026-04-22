import numpy as np
from .base import Layer

class AvgPool1D(Layer):
  def __init__(self, kernel_size, stride=None, padding=0):
    self.kernel_size = kernel_size
    self.stride = stride if stride is not None else kernel_size
    self.padding = padding
  
  def forward(self, x):
    x = np.asarray(x)
    self.x = x

    batch_size, channels, input_len = x.shape

    if self.padding > 0:
      x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding)), constant_values=0)
    else:
      x_padded = x

    self.x_padded = x_padded

    out_len = (input_len + 2 * self.padding - self.kernel_size) // self.stride + 1

    out = np.zeros((batch_size, channels, out_len))

    for b in range(batch_size):
      for c in range(channels):
        for i in range(out_len):
          start = i * self.stride
          end = start + self.kernel_size

          window = x_padded[b, c, start:end]

          out[b, c, i] = np.mean(window)

    return out
  
  def backward(self, grad):
    grad = np.asarray(grad)

    batch_size, channels, out_len = grad.shape

    dx_padded = np.zeros_like(self.x_padded)

    for b in range(batch_size):
      for c in range(channels):
        for i in range(out_len):
          start = i * self.stride
          end = start + self.kernel_size

          dx_padded[b, c, start:end] += grad[b, c, i] / self.kernel_size

    if self.padding > 0:
      dx = dx_padded[:, :, self.padding:-self.padding]
    else:
      dx = dx_padded

    return dx
  
  def parameters(self):
    return []
  
  def get_config(self):
    return {
      "type": "AvgPool1D",
      "kernel_size": self.kernel_size,
      "stride": self.stride,
      "padding": self.padding
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
  
class AvgPool2D(Layer):
  def __init__(self, kernel_size, stride=None, padding=0):
    self.kernel_size = kernel_size
    self.stride = stride if stride is not None else kernel_size
    self.padding = padding
  
  def forward(self, x):
    x = np.asarray(x)
    self.x = x

    batch_size, channels, height, width = x.shape

    if self.padding > 0:
      x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), constant_values=0)
    else:
      x_padded = x

    self.x_padded = x_padded

    out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
    out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

    out = np.zeros((batch_size, channels, out_height, out_width))

    for b in range(batch_size):
      for c in range(channels):
        for i in range(out_height):
          for j in range(out_width):
            h_start = i * self.stride
            h_end = h_start + self.kernel_size

            w_start = j * self.stride
            w_end = w_start + self.kernel_size

            window = x_padded[b, c, h_start:h_end, w_start:w_end]

            out[b, c, i, j] = np.mean(window)

    return out
  
  def backward(self, grad):
    grad = np.asarray(grad)

    batch_size, channels, out_height, out_width = grad.shape

    dx_padded = np.zeros_like(self.x_padded)

    for b in range(batch_size):
      for c in range(channels):
        for i in range(out_height):
          for j in range(out_width):
            h_start = i * self.stride
            h_end = h_start + self.kernel_size

            w_start = j * self.stride
            w_end = w_start + self.kernel_size

            dx_padded[b, c, h_start:h_end, w_start:w_end] += grad[b, c, i, j] / (self.kernel_size * self.kernel_size)

    if self.padding > 0:
      dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
    else:
      dx = dx_padded

    return dx
  
  def parameters(self):
    return []
  
  def get_config(self):
    return {
      "type": "AvgPool2D",
      "kernel_size": self.kernel_size,
      "stride": self.stride,
      "padding": self.padding
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
  
class AvgPool3D(Layer):
  def __init__(self, kernel_size, stride=None, padding=0):
    self.kernel_size = kernel_size
    self.stride = stride if stride is not None else kernel_size
    self.padding = padding

  def forward(self, x):
    x = np.asarray(x)
    self.x = x

    batch_size, channels, depth, height, width = x.shape

    if self.padding > 0:
      x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding)), constant_values=0)
    else:
      x_padded = x

    self.x_padded = x_padded

    out_depth = (depth + 2 * self.padding - self.kernel_size) // self.stride + 1
    out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
    out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

    out = np.zeros((batch_size, channels, out_depth, out_height, out_width))

    for b in range(batch_size):
      for c in range(channels):
        for d in range(out_depth):
          for i in range(out_height):
            for j in range(out_width):
              d_start = d * self.stride
              d_end = d_start + self.kernel_size

              h_start = i * self.stride
              h_end = h_start + self.kernel_size

              w_start = j * self.stride
              w_end = w_start + self.kernel_size

              window = x_padded[b, c, d_start:d_end, h_start:h_end, w_start:w_end]

              out[b, c, d, i, j] = np.mean(window)

    return out
  
  def backward(self, grad):
    grad = np.asarray(grad)

    batch_size, channels, out_depth, out_height, out_width = grad.shape

    dx_padded = np.zeros_like(self.x_padded)

    for b in range(batch_size):
      for c in range(channels):
        for d in range(out_depth):
          for i in range(out_height):
            for j in range(out_width):
              d_start = d * self.stride
              d_end = d_start + self.kernel_size

              h_start = i * self.stride
              h_end = h_start + self.kernel_size

              w_start = j * self.stride
              w_end = w_start + self.kernel_size

              dx_padded[b, c, d_start:d_end, h_start:h_end, w_start:w_end] += grad[b, c, d, i, j] / (self.kernel_size * self.kernel_size * self.kernel_size)

    if self.padding > 0:
      dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding, self.padding:-self.padding]
    else:
      dx = dx_padded

    return dx
  
  def parameters(self):
    return []
  
  def get_config(self):
    return {
      "type": "AvgPool3D",
      "kernel_size": self.kernel_size,
      "stride": self.stride,
      "padding": self.padding
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"