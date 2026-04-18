import numpy as np

def argmax(x, axis=None):
  x = np.asarray(x)
  if axis is None:
    flat = x.ravel()
    max_idx = 0
    for i in range(1, flat.size):
      if flat[i] > flat[max_idx]:
        max_idx = i
    return np.unravel_index(max_idx, x.shape) if x.ndim > 1 else max_idx

  # move target axis to the end
  x = np.moveaxis(x, axis, -1)
  out_shape = x.shape[:-1]
  x_flat = x.reshape(-1, x.shape[-1])

  result = np.empty(x_flat.shape[0], dtype=int)
  for i in range(x_flat.shape[0]):
    max_idx = 0
    for j in range(1, x_flat.shape[1]):
      if x_flat[i, j] > x_flat[i, max_idx]:
        max_idx = j
    result[i] = max_idx

  return result.reshape(out_shape)