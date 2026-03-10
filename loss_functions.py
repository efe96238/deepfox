import numpy as np

def MSE(y, y_pred):
  return np.mean((y - y_pred) ** 2)

def BinaryCE(y, y_pred):
  e = 1e-15
  y_pred = np.clip(y_pred, e, 1 - e) #to prevent log(0) = -inf, NaN
  return -np.mean((y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))

def ClassCE(y, y_pred):
  e = 1e-15
  y_pred = np.clip(y_pred, e, 1 - e) #to prevent log(0) = -inf, NaN
  return -np.mean(np.sum(y * np.log(y_pred), axis=1))