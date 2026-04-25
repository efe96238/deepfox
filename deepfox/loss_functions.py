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