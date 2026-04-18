import json
import zipfile
import numpy as np


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

  def _build_manifest(self):
    params = self.parameters()

    manifest = {
      "format": "DeepFox",
      "version": 1,
      "layers": [layer.get_config() for layer in self.layers],
      "parameters": []
    }

    for i, p in enumerate(params):
      manifest["parameters"].append({
        "name": f"param_{i}",
        "shape": list(p.data.shape),
        "dtype": str(p.data.dtype)
      })

    return manifest

  def save(self, path):
    if not path.endswith(".dpx"):
      path += ".dpx"

    manifest = self._build_manifest()
    params = self.parameters()

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
      archive.writestr("manifest.json", json.dumps(manifest, indent=2))

      for i, p in enumerate(params):
        with archive.open(f"param_{i}.npy", "w") as f:
          np.save(f, p.data)

  def load(self, path):
    with zipfile.ZipFile(path, "r") as archive:
      manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
      params = self.parameters()

      if manifest.get("format") != "DeepFox":
        raise ValueError("Invalid .dpx file format.")

      saved_layers = manifest.get("layers", [])
      current_layers = [layer.get_config() for layer in self.layers]

      if saved_layers != current_layers:
        raise ValueError("Model architecture does not match the .dpx file.")

      saved_params = manifest.get("parameters", [])

      if len(saved_params) != len(params):
        raise ValueError("Parameter count does not match the .dpx file.")

      for i, p in enumerate(params):
        with archive.open(f"param_{i}.npy") as f:
          loaded = np.load(f)

        expected_shape = tuple(saved_params[i]["shape"])

        if loaded.shape != expected_shape:
          raise ValueError(f"Shape mismatch for param_{i}: expected {expected_shape}, got {loaded.shape}")

        p.data[...] = loaded
        p.grad = np.zeros_like(p.data)

  def __call__(self, x):
    return self.forward(x)