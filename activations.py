import numpy as np

def ReLu(x):
  return np.maximum(0, x)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def tanh(x):
  return np.tanh(x)

def softmax(x):
  e = np.exp(x - np.max(x))
  return e / np.sum(e)