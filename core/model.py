
class Model:
  def __init__(self, *layers):
    self.layers = list(layers)

  def add(self, layer):
    self.layers.append(layer)

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

  def zero_grad(self):
    for p in self.parameters():
      p.zero_grad()

  def __call__(self, x):
    return self.forward(x)