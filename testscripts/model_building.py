import numpy as np
import deepfox as dfx
from deepfox import Linear, ReLU, Sequential

data = np.genfromtxt("testsets/non_linear_dataset.csv", delimiter=",", skip_header=1)
X = data[:, 0:1]
y = data[:, 1:2]

print("=== Pattern 1: Single Sequential ===")
model1 = dfx.Model(Sequential(Linear(1, 16), ReLU(), Linear(16, 16), ReLU(), Linear(16, 1)))

criterion = dfx.MSE()
optimizer = dfx.Adam(lr=0.01)

print(f"Parameter count: {len(model1.parameters())}")

for epoch in range(3000):
  pred = model1(X)
  loss = criterion.forward(y, pred)
  grad = criterion.backward()
  model1.zero_grad()
  model1.backward(grad)
  optimizer.step(model1.parameters())

  if epoch % 500 == 0:
      print(f"  epoch={epoch}, loss={loss:.6f}")

print(f"  Final loss: {loss:.6f}")

print("\n=== Pattern 2: Multiple Blocks ===")
model2 = dfx.Model(
    Sequential(Linear(1, 16), ReLU()),
    Sequential(Linear(16, 16), ReLU()),
    Sequential(Linear(16, 1))
)

criterion2 = dfx.MSE()
optimizer2 = dfx.Adam(lr=0.01)

print(f"Parameter count: {len(model2.parameters())}")

for epoch in range(3000):
    pred = model2(X)
    loss2 = criterion2.forward(y, pred)
    grad = criterion2.backward()
    model2.zero_grad()
    model2.backward(grad)
    optimizer2.step(model2.parameters())

    if epoch % 500 == 0:
        print(f"  epoch={epoch}, loss={loss2:.6f}")

print(f"  Final loss: {loss2:.6f}")

# --- Verify parameter counts match ---
print(f"\n=== Checks ===")
print(f"Model 1 params: {len(model1.parameters())}")
print(f"Model 2 params: {len(model2.parameters())}")
print(f"Param counts equal: {len(model1.parameters()) == len(model2.parameters())}")