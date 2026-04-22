"""RepQ Relay model definitions.

Canonical modules use the generic names `builder`, `layers`, `vit`, and `swin`.
Legacy `build_repq_model` / `repq_*` names remain as aliases for compatibility.
"""

from . import builder, layers, swin, vit

build_repq_model = builder
repq_layers = layers
repq_swin = swin
repq_vit = vit

__all__ = [
    "builder",
    "layers",
    "vit",
    "swin",
    "build_repq_model",
    "repq_layers",
    "repq_swin",
    "repq_vit",
]
