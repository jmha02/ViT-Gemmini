"""Top-level model package grouped by family.

Use `models.ivit` for I-ViT Relay definitions and `models.repq` for RepQ Relay definitions.
Legacy module names are exposed as import aliases for compatibility.
"""

import sys as _sys

from . import ivit, repq
from .ivit import builder as build_model
from .ivit import layers as quantized_layers
from .ivit import swin as quantized_swin
from .ivit import utils
from .ivit import vit as quantized_vit
from .repq import builder as build_repq_model
from .repq import layers as repq_layers
from .repq import swin as repq_swin
from .repq import vit as repq_vit

_SUBMODULE_ALIASES = {
    "build_model": build_model,
    "quantized_layers": quantized_layers,
    "quantized_swin": quantized_swin,
    "quantized_vit": quantized_vit,
    "utils": utils,
    "build_repq_model": build_repq_model,
    "repq_layers": repq_layers,
    "repq_swin": repq_swin,
    "repq_vit": repq_vit,
}

for _name, _module in _SUBMODULE_ALIASES.items():
    globals()[_name] = _module
    _sys.modules[f"{__name__}.{_name}"] = _module

__all__ = [
    "ivit",
    "repq",
    *sorted(_SUBMODULE_ALIASES),
]
