import numpy as np

import deepfox as dfx
from deepfox import Linear

data = np.genfromtxt("testsets/linear_regression_dataset.csv", delimiter=",", skip_header=1)

X = data[:, 0:1]
y = data[:, 1:2]

model = dfx.Model(Linear(1, 1))
criterion = dfx.MSE()
optimizer = dfx.AdamW(lr=0.01)

epochs = 3000

for epoch in range(epochs):
  pred = model(X)

  loss = criterion.forward(y, pred)

  grad = criterion.backward()

  model.zero_grad()
  model.backward(grad)

  optimizer.step(model.parameters())

  if epoch % 100 == 0:
    print(f"epoch={epoch}, loss={loss:.6f}")

final_pred = model(X)
final_loss = criterion.forward(y, final_pred)

linear_layer = model.blocks[0]
print("\nTraining finished.")
print(f"Final loss: {final_loss:.6f}")
print("Learned weights:", linear_layer.weights.data)
print("Learned bias:", linear_layer.bias.data)

print("\nSample predictions:")
for i in range(5):
  print(f"x={X[i,0]:.2f}, y_true={y[i,0]:.4f}, y_pred={final_pred[i,0]:.4f}")