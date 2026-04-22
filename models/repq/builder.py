from __future__ import annotations

import numpy as np
import tvm
from tvm import relay

from .layers import RepQContext, RepQModuleMeta
from .swin import get_repq_swin_tiny_model
from .vit import get_repq_deit_tiny_model


def _resolve_data_shape(batch_size, image_shape, data_layout):
    if data_layout == "NCHW":
        return (batch_size,) + image_shape
    raise RuntimeError(f"Unsupported data layout for RepQ Relay builder: {data_layout}")


def _build_function(name: str, data_shape, debug_unit: str | None = None):
    if name == "deit_tiny_patch16_224":
        return get_repq_deit_tiny_model(data_shape, debug_unit=debug_unit)
    if name == "swin_tiny_patch4_window7_224":
        return get_repq_swin_tiny_model(data_shape, debug_unit=debug_unit)
    raise RuntimeError(f"Unsupported RepQ model: {name}")


def _filter_params(mod: tvm.IRModule, params: dict[str, np.ndarray]) -> dict[str, tvm.nd.NDArray]:
    expected = {var.name_hint for var in mod["main"].params if var.name_hint != "data"}
    missing = sorted(expected - params.keys())
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(f"Missing RepQ Relay parameters: {preview}")

    filtered = {}
    for name in expected:
        array = params[name]
        filtered[name] = tvm.nd.array(array)
    return filtered


def get_workload(
    name: str,
    params: dict[str, np.ndarray],
    meta: dict[str, RepQModuleMeta],
    batch_size: int = 1,
    image_shape=(3, 224, 224),
    data_layout: str = "NCHW",
    debug_unit: str | None = None,
):
    if batch_size != 1:
        raise RuntimeError("The RepQ TVM path currently supports batch_size = 1 only.")

    data_shape = _resolve_data_shape(batch_size, image_shape, data_layout)
    RepQContext.set_artifacts(meta, params)
    net = _build_function(name, data_shape, debug_unit=debug_unit)
    mod = tvm.IRModule.from_expr(net)
    mod = relay.transform.InferType()(mod)
    return mod, _filter_params(mod, params)
