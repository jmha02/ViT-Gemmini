from .vit import Q_VisionTransformer
from .ptq4_deit_relay import PTQ4_VisionTransformer
from .ptq4_checkpoint import (
    build_ptq4_param_dict,
    load_ptq4_state_dict,
    random_ptq4_state_dict,
    DEFAULT_PTQ4_DEIT_T_CHECKPOINT,
)
from .swin import get_swin_small_model, get_swin_tiny_model
from .utils import create_workload, QuantizeInitializer

import tvm
from tvm import relay


def get_deit(name,
            batch_size,
            image_shape=(3, 224, 224),
            dtype="int8",
            data_layout="NCHW",
            kernel_layout="OIHW",
            debug_unit=None):


    if data_layout == 'NCHW':
        data_shape = (batch_size,) + image_shape
    elif data_layout == 'NHWC':
        data_shape = (batch_size, image_shape[1], image_shape[2], image_shape[0])
    elif data_layout == 'HWCN':
        data_shape = (image_shape[1], image_shape[2], image_shape[0], batch_size)
    elif data_layout == 'HWNC':
        data_shape = (image_shape[1], image_shape[2], batch_size, image_shape[0])
    else:
        raise RuntimeError("Unsupported data layout {}".format(data_layout))


    if name == 'deit_tiny_patch16_224':
        embed_dim = 192
        num_heads = 3
    elif name == 'deit_small_patch16_224':
        embed_dim = 384
        num_heads = 6
    elif name == 'deit_base_patch16_224':
        embed_dim = 768
        num_heads = 12
    else:
        raise RuntimeError("Unsupported model {}".format(name))
    

    return Q_VisionTransformer(
        data_shape=data_shape,
        dtype=dtype,
        patch_size=16,
        num_patches=196,
        in_chans=3,
        num_classes=1000,
        embed_dim=embed_dim,
        depth=12,
        num_heads=num_heads,
        mlp_ratio=4,
        debug_unit=debug_unit,
    )

def get_ptq4_deit(
    name,
    batch_size,
    image_shape=(3, 224, 224),
    data_layout="NCHW",
    debug_unit=None,
):
    if data_layout == "NCHW":
        data_shape = (batch_size,) + image_shape
    elif data_layout == "NHWC":
        data_shape = (batch_size, image_shape[1], image_shape[2], image_shape[0])
    else:
        raise RuntimeError("Unsupported data layout {}".format(data_layout))

    if name == "ptq4_deit_tiny_patch16_224":
        embed_dim, num_heads = 192, 3
    elif name == "ptq4_deit_small_patch16_224":
        embed_dim, num_heads = 384, 6
    else:
        raise RuntimeError("Unsupported PTQ4 model {}".format(name))

    return PTQ4_VisionTransformer(
        data_shape=data_shape,
        embed_dim=embed_dim,
        depth=12,
        num_heads=num_heads,
        mlp_ratio=4,
        debug_unit=debug_unit,
    )


def get_swin(
            name,
            batch_size,
            image_shape=(3, 224, 224),
            dtype="int8",
            data_layout="NCHW",
            kernel_layout="OIHW",
            debug_unit=None):


    if data_layout == 'NCHW':
        data_shape = (batch_size,) + image_shape
    elif data_layout == 'NHWC':
        data_shape = (batch_size, image_shape[1], image_shape[2], image_shape[0])
    elif data_layout == 'HWCN':
        data_shape = (image_shape[1], image_shape[2], image_shape[0], batch_size)
    elif data_layout == 'HWNC':
        data_shape = (image_shape[1], image_shape[2], batch_size, image_shape[0])
    else:
        raise RuntimeError("Unsupported data layout {}".format(data_layout))

    if name == "swin_tiny_patch4_window7_224":
        return get_swin_tiny_model(
            data_shape=data_shape,
            dtype=dtype,
            debug_unit=debug_unit,
        )

    raise RuntimeError("Unsupported model {}".format(name))


def get_workload(name,
                 batch_size=1,
                 image_shape=(3, 224, 224),
                 dtype="int8",
                 data_layout="NCHW",
                 kernel_layout="OIHW",
                 debug_unit=None):

    if batch_size != 1:
        raise RuntimeError("The released project only supports batch_size = 1.")

    if name.startswith("ptq4_deit_"):
        net = get_ptq4_deit(
            name,
            batch_size,
            image_shape=image_shape,
            data_layout=data_layout,
            debug_unit=debug_unit,
        )
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)
        if name == "ptq4_deit_tiny_patch16_224" and DEFAULT_PTQ4_DEIT_T_CHECKPOINT.is_file():
            state_dict = load_ptq4_state_dict(DEFAULT_PTQ4_DEIT_T_CHECKPOINT)
        elif name == "ptq4_deit_small_patch16_224":
            state_dict = random_ptq4_state_dict(embed_dim=384, num_heads=6)
        else:
            state_dict = random_ptq4_state_dict(embed_dim=192, num_heads=3)
        param_np = build_ptq4_param_dict(state_dict)
        params = {}
        for param in mod["main"].params:
            key = param.name_hint
            if key == "data":
                continue
            if key not in param_np:
                raise KeyError(f"Missing PTQ4 checkpoint param for relay var: {key}")
            params[key] = tvm.nd.array(param_np[key])
        return mod, params
    elif name.startswith("deit_"):
        net = get_deit(name,
                    batch_size,
                    image_shape=image_shape,
                    dtype=dtype,
                    data_layout=data_layout,
                    kernel_layout=kernel_layout,
                    debug_unit=debug_unit)
    elif name.startswith("swin_"):
        net = get_swin(name,
                    batch_size,
                    image_shape=image_shape,
                    dtype=dtype,
                    data_layout=data_layout,
                    kernel_layout=kernel_layout,
                    debug_unit=debug_unit)
    else:
        raise RuntimeError("Unsupported model {}".format(name))

    return create_workload(net, QuantizeInitializer())
