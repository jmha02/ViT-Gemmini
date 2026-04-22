"""I-ViT Relay model definitions.

Canonical modules use the generic names `builder`, `layers`, `vit`, and `swin`.
Legacy `build_model` / `quantized_*` names remain as aliases for compatibility.
"""

from . import builder, layers, swin, utils, vit

build_model = builder
quantized_layers = layers
quantized_swin = swin
quantized_vit = vit

__all__ = [
    "builder",
    "layers",
    "vit",
    "swin",
    "utils",
    "build_model",
    "quantized_layers",
    "quantized_swin",
    "quantized_vit",
]
