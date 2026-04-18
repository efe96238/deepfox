import numpy as np
import os
import deepfox as dfx
from deepfox import Linear, ReLU

data = np.genfromtxt("testsets/non_linear_dataset.csv", delimiter=",", skip_header=1)

X = data[:, 0:1]
y = data[:, 1:2]

model = dfx.Model(Linear(1, 16), ReLU(), Linear(16, 16), ReLU(), Linear(16, 1))
criterion = dfx.MSE()
optimizer = dfx.Adam(lr=0.01)

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

os.makedirs("models", exist_ok=True)
model.save("models/save_test.dpx")

linear_layer = model.layers[0]
print("\nTraining finished.")
print(f"Final loss: {final_loss:.6f}")
print("Learned weights:", linear_layer.weights.data)
print("Learned bias:", linear_layer.bias.data)

print("\nSample predictions:")
for i in range(5):
  print(f"x={X[i,0]:.2f}, y_true={y[i,0]:.4f}, y_pred={final_pred[i,0]:.4f}")