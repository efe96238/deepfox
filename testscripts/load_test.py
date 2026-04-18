import numpy as np

import deepfox as dfx
from deepfox import Linear, ReLU

data = np.genfromtxt("testsets/non_linear_dataset.csv", delimiter=",", skip_header=1)

X = data[:, 0:1]
y = data[:, 1:2]

model = dfx.Model(Linear(1, 16), ReLU(), Linear(16, 16), ReLU(), Linear(16, 1))

model.load("models/save_test.dpx")

test_pred = model(X)

print("\nSample predictions:")
for i in range(5):
  print(f"x={X[i,0]:.2f}, y_true={y[i,0]:.4f}, y_pred={test_pred[i,0]:.4f}")