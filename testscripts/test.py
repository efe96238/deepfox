import deepfox as dfx
from deepfox import Linear, ReLU, Conv2D, Sequential

# --- Test 1: Default mode ---
print("=== Test 1: Default Mode ===")
model = dfx.Model(
  Sequential(Linear(1, 16), ReLU(), Linear(16, 1))
)
print(f"Model training: {model.training}")
print(f"Sequential training: {model.blocks[0].training}")
print(f"Linear training: {model.blocks[0].layers[0].training}")
print(f"ReLU training: {model.blocks[0].layers[1].training}")
print(f"All True: {model.training and model.blocks[0].training and model.blocks[0].layers[0].training}")

# --- Test 2: Eval mode ---
print("\n=== Test 2: Eval Mode ===")
model.eval()
print(f"Model training: {model.training}")
print(f"Sequential training: {model.blocks[0].training}")
print(f"Linear training: {model.blocks[0].layers[0].training}")
print(f"ReLU training: {model.blocks[0].layers[1].training}")
print(f"All False: {not model.training and not model.blocks[0].training and not model.blocks[0].layers[0].training}")

# --- Test 3: Back to train mode ---
print("\n=== Test 3: Back to Train ===")
model.train()
print(f"Model training: {model.training}")
print(f"Sequential training: {model.blocks[0].training}")
print(f"Linear training: {model.blocks[0].layers[0].training}")
print(f"All True: {model.training and model.blocks[0].training and model.blocks[0].layers[0].training}")

# --- Test 4: Multi-block model ---
print("\n=== Test 4: Multi-Block ===")
model2 = dfx.Model(
  Sequential(Linear(1, 16), ReLU()),
  Sequential(Linear(16, 16), ReLU()),
  Sequential(Linear(16, 1))
)
model2.eval()
all_eval = all(
  layer.training == False
  for block in model2.blocks
  for layer in block.layers
)
print(f"All layers in eval: {all_eval}")

model2.train()
all_train = all(
  layer.training == True
  for block in model2.blocks
  for layer in block.layers
)
print(f"All layers in train: {all_train}")

# --- Test 5: Chaining ---
print("\n=== Test 5: Chaining ===")
result = model.eval()
print(f"Chaining works: {result is model}")

# --- Test 6: Flat blocks ---
print("\n=== Test 6: Flat Blocks ===")
model3 = dfx.Model(Linear(1, 16), ReLU(), Linear(16, 1))
model3.eval()
print(f"Linear training: {model3.blocks[0].training}")
print(f"ReLU training: {model3.blocks[1].training}")
print(f"All False: {not model3.blocks[0].training and not model3.blocks[1].training and not model3.blocks[2].training}")