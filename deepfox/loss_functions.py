import numpy as np

class MSE:
  def forward(self, y, y_pred):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    self.y = y
    self.y_pred = y_pred

    return np.mean((y - y_pred) ** 2)

  def backward(self):
    n = self.y.size
    return (2 / n) * (self.y_pred - self.y)
  
class MAE:
  def forward(self, y, y_pred):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    self.y = y
    self.y_pred = y_pred

    return np.mean(np.abs(y - y_pred))
  
  def backward(self):
    n = self.y.size
    return np.sign(self.y_pred - self.y) / n

class BCE:
  def forward(self, y, y_pred):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1.0 - eps) #so that log(0) does not return -inf or NaN

    self.y = y
    self.y_pred = y_pred

    return -np.mean(y * np.log(y_pred) + (1.0 - y) * np.log(1.0 - y_pred))

  def backward(self):
    n = self.y.size
    return (-(self.y / self.y_pred) + (1 - self.y) / (1 - self.y_pred)) / n
  
class BCEWithLogits:
  def forward(self, y, logits):
    y = np.asarray(y)
    logits = np.asarray(logits)
    self.y = y

    self.sigmoid = np.where(logits >= 0, 1 / (1 + np.exp(-logits)), np.exp(logits) / (1 + np.exp(logits)))
    return np.mean(np.maximum(logits, 0) + (-logits * y) + np.log(1 + np.exp(-np.abs(logits))))
  
  def backward(self):
    n = self.y.size
    return (self.sigmoid - self.y) / n

class CrossEntropy:
  def forward(self, y, y_pred):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1.0 - eps) #so that log(0) does not return -inf or NaN

    self.y = y
    self.y_pred = y_pred

    return -np.mean(np.sum(y * np.log(y_pred), axis=1))

  def backward(self):
    batch_size = self.y.shape[0]
    return -(self.y / self.y_pred) / batch_size
  
class CrossEntropyWithLogits:
  def forward(self, y, logits):
    y = np.asarray(y)
    logits = np.asarray(logits)

    self.y = y

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    log_sum_exp = np.log(np.sum(exp_shifted, axis=1, keepdims=True))
    log_softmax = shifted - log_sum_exp

    self.softmax = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    return -np.mean(np.sum(y * log_softmax, axis=1))

  def backward(self):
    batch_size = self.y.shape[0]
    return (self.softmax - self.y) / batch_size
  
class HuberLoss:
  def __init__(self, delta=1.0):
    self.delta = delta

  def forward(self, y, y_pred):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    self.y = y
    self.y_pred = y_pred

    return np.mean(np.where(np.abs(y - y_pred) <= self.delta, 0.5 * (y - y_pred)**2, self.delta * (np.abs(y - y_pred) - 0.5 * self.delta)))
  
  def backward(self):
    n = self.y.size
    return np.where(np.abs(self.y - self.y_pred) <= self.delta, (self.y_pred - self.y) / n, self.delta * np.sign(self.y_pred - self.y) / n)
  
class NLLLoss:
  def forward(self, y, log_probs):
    y = np.asarray(y)
    log_probs = np.asarray(log_probs)

    self.y = y

    return -np.mean(np.sum(y * log_probs, axis=1))
  
  def backward(self):
    batch_size = self.y.shape[0]
    return -self.y / batch_size
  
class HingeLoss:
  def forward(self, y, y_pred):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    self.y = y
    self.y_pred = y_pred

    return np.mean(np.maximum(0, 1 - y * y_pred))
  
  def backward(self):
    n = self.y.size
    return np.where(1 - self.y * self.y_pred > 0, -self.y / n, 0)
  
class KLDivergence:
  def forward(self, y, log_probs):
    y = np.asarray(y)
    log_probs = np.asarray(log_probs)

    eps = 1e-15
    y = np.clip(y, eps, 1.0 - eps)

    self.y = y

    return np.mean(np.sum(y * (np.log(y) - log_probs), axis=1))
  
  def backward(self):
    batch_size = self.y.shape[0]
    return -self.y / batch_size
  
class CosineEmbeddingLoss:
  def __init__(self, margin=0.0):
    self.margin = margin

  def forward(self, y, x1, x2):
    y = np.asarray(y, dtype=np.float64)
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    self.y = y
    self.x1 = x1
    self.x2 = x2

    eps = 1e-8

    self.dot = np.sum(x1 * x2, axis=1, keepdims=True)
    self.norm1 = np.sqrt(np.sum(x1 ** 2, axis=1, keepdims=True)) + eps
    self.norm2 = np.sqrt(np.sum(x2 ** 2, axis=1, keepdims=True)) + eps

    self.cos = self.dot / (self.norm1 * self.norm2)

    loss = np.where(y == 1, 1 - self.cos, np.maximum(0, self.cos - self.margin))

    return np.mean(loss)

  def backward(self):
    batch_size = self.y.shape[0]

    dcos_dx1 = (self.x2 / (self.norm1 * self.norm2)) - self.cos * (self.x1 / (self.norm1 ** 2))
    dcos_dx2 = (self.x1 / (self.norm1 * self.norm2)) - self.cos * (self.x2 / (self.norm2 ** 2))

    mask_pos = (self.y == 1)
    mask_neg = (self.y == -1) & (self.cos - self.margin > 0)

    scale = np.where(mask_pos, -1.0, np.where(mask_neg, 1.0, 0.0))

    dx1 = scale * dcos_dx1 / batch_size
    dx2 = scale * dcos_dx2 / batch_size

    return dx1, dx2
  
class SmoothL1Loss:
  def __init__(self, beta=1.0):
    self.beta = beta

  def forward(self, y, y_pred):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    self.y = y
    self.y_pred = y_pred

    return np.mean(np.where(np.abs(y - y_pred) <= self.beta, 0.5 * (y - y_pred)**2 / self.beta, np.abs(y - y_pred) - 0.5 * self.beta))

  def backward(self):
    n = self.y.size
    return np.where(np.abs(self.y - self.y_pred) <= self.beta, (self.y_pred - self.y) / (self.beta * n), np.sign(self.y_pred - self.y) / n)