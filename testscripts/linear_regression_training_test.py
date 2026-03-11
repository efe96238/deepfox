import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from core.layers import Linear
from core.model import Model
from core.loss_functions import MSE
from core.optimizers import Adam

data = np.genfromtxt("testsets/linear_regression_dataset.csv", delimiter=",", skip_header=1)

X = data[:, 0:1]
y = data[:, 1:2]

model = Model(Linear(1, 1))
criterion = MSE()
optimizer = Adam(lr=0.03)

epochs = 3000

for epoch in range(epochs):
  pred = model(X)

  loss = criterion.forward(y, pred)

  grad = criterion.backward()

  model.zero_grad()
  model.backward(grad)

  optimizer.step(model.parameters(), model.grads())

  if epoch % 100 == 0:
    print(f"epoch={epoch}, loss={loss:.6f}")

final_pred = model(X)
final_loss = criterion.forward(y, final_pred)

linear_layer = model.layers[0]
print("\nTraining finished.")
print(f"Final loss: {final_loss:.6f}")
print("Learned weights:", linear_layer.weights)
print("Learned bias:", linear_layer.bias)

print("\nSample predictions:")
for i in range(5):
  print(f"x={X[i,0]:.2f}, y_true={y[i,0]:.4f}, y_pred={final_pred[i,0]:.4f}")