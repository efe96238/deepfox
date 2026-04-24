# DeepFox

A lightweight deep learning framework built from scratch with NumPy. DeepFox aims to be as low-level and customizable as PyTorch, while offering the simplicity of scikit-learn's API.

## Installation

install from source:

```bash
git clone https://github.com/efe96238/deepfox.git
cd deepfox
pip install -e .
```

## Quick Start

```python
import numpy as np
import deepfox as dfx
from deepfox import Linear, ReLU, Sequential

# Build a model
model = dfx.Model(
    Sequential(Linear(1, 16), ReLU(), Linear(16, 1))
)

# Or use multiple blocks
model = dfx.Model(
    Sequential(Linear(1, 16), ReLU()),
    Sequential(Linear(16, 16), ReLU()),
    Sequential(Linear(16, 1))
)

# Train
criterion = dfx.MSE()
optimizer = dfx.Adam(lr=0.01)

for epoch in range(1000):
    pred = model(X)
    loss = criterion.forward(y, pred)
    grad = criterion.backward()
    model.zero_grad()
    model.backward(grad)
    optimizer.step(model.parameters())
```

## What's Included

**Layers**
- `Linear` — fully connected layer
- `Conv1D`, `Conv2D`, `Conv3D` — convolutional layers
- `MaxPool1D`, `MaxPool2D`, `MaxPool3D` — max pooling
- `AvgPool1D`, `AvgPool2D` — average pooling
- `AdaptiveAvgPool1D`, `AdaptiveAvgPool2D`, `AdaptiveAvgPool3D` — adaptive average pooling
- `Sequential` — sequential layer container
- `BatchNorm1D`, `BatchNorm2D`, `BatchNorm3D` - batch normalization
- `Dropout`
- `Flatten`

**Activations**
- `ReLU`, `Sigmoid`, `Tanh`, `Softmax`

**Loss Functions**
- `MSE`, `MAE`, `BinaryCE`, `ClassCE`

**Optimizers**
- `Adam`, `AdamW`, `SGD`, `MomentumSGD`, `RMSProp`

**Model Features**
- Save and load models with `.dpx` format
- Architecture validation on load
- Clean `repr` for model inspection
- Composable block-based model building

## Save & Load

```python
# Save
model.save("my_model.dpx")

# Load (architecture must match)
model.load("my_model.dpx")
```

## Requirements

- Python 3.8+
- NumPy

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
