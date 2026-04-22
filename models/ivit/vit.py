import re

import tvm
from tvm import relay

from . import layers


_ONLY_DEIT_BLOCK_RE = re.compile(r"^only_block(\d+)$")


def _parse_only_deit_block(debug_unit, depth):
    if not debug_unit:
        return None
    match = _ONLY_DEIT_BLOCK_RE.match(debug_unit)
    if not match:
        return None
    block_idx = int(match.group(1))
    if not 0 <= block_idx < depth:
        raise RuntimeError(f"Unsupported DeiT standalone block index: {block_idx}")
    return block_idx


def Q_Block(
    data,
    name,
    dim,
    num_heads,
    mlp_ratio,
    qk_scale,
    batch_size,
    rounding="TRUNCATE",
    debug_unit=None,
):
    "Attention mudule"
    shortcut = data

    ## layer_norm
    qconfig0 = layers.get_qconfig(name + "_qconfig_norm1")
    norm1_bias = relay.var(name + "_norm1_bias", shape=[dim], dtype="int64")
    norm1 = layers.quantized_layernorm(data, norm1_bias)
    if debug_unit == f"{name}_norm1":
        return norm1

    ## attention
    qconfig1 = layers.get_qconfig(name + "_qconfig_qkv")
    req1 = layers.requantize(
        norm1,
        input_scale=qconfig0.output_scale,
        output_scale=qconfig1.input_scale,
        out_dtype=qconfig1.input_dtype,
        force_float=True,
    )

    req1 = relay.reshape(req1, [-3, 0])
    qkv = layers.quantized_dense(
        data=req1,
        name=name + "_attn_qkv",
        input_scale=qconfig1.input_scale,
        kernel_scale=qconfig1.kernel_scale,
        units=dim * 3,
        kernel_shape=(dim * 3, dim),
        kernel_dtype="int8",
        add_bias=True,
    )
    qconfig2 = layers.get_qconfig(name + "_qconfig_matmul_1")
    req2 = layers.requantize(
        qkv,
        input_scale=qconfig1.output_scale,
        output_scale=qconfig2.input_scale,
        out_dtype=qconfig2.input_dtype,
    )

    qkv = relay.reshape(req2, [-4, batch_size, -1, -2])
    qkv_reshape = relay.reshape(qkv, [0, 0, 3, num_heads, -1])
    qkv = relay.transpose(qkv_reshape, [2, 0, 3, 1, 4])
    qkv = relay.split(qkv, 3, axis=0)
    q = relay.reshape(relay.squeeze(qkv[0], axis=[0]), [-3, -2])
    k = relay.reshape(relay.squeeze(qkv[1], axis=[0]), [-3, -2])
    v = relay.reshape(relay.squeeze(qkv[2], axis=[0]), [-3, -2])

    qconfig3 = layers.get_qconfig(name + "_qconfig_softmax")
    req3 = layers.quantized_matmul_via_dense(
        q,
        k,
        input_scale1=qconfig2.input_scale,
        input_scale2=qconfig2.input_scale,
        requant_input_scale=qconfig2.output_scale * qk_scale,
        requant_output_scale=qconfig3.input_scale,
        requant_out_dtype=qconfig3.input_dtype,
    )
    req3 = relay.reshape(req3, [-4, -1, num_heads, -2])

    if debug_unit == f"{name}_pre_softmax":
        return req3

    attn = layers.quantized_softmax(req3, qconfig3.input_scale)
    if debug_unit == f"{name}_post_softmax":
        return attn

    qconfig4 = layers.get_qconfig(name + "_qconfig_matmul_2")
    qconfig5 = layers.get_qconfig(name + "_qconfig_proj")
    attn = relay.reshape(attn, [-3, -2])
    v = relay.transpose(v, [0, 2, 1])
    attn = layers.quantized_matmul_via_dense(
        attn,
        v,
        input_scale1=qconfig4.input_scale,
        input_scale2=qconfig2.input_scale,
        requant_input_scale=qconfig4.output_scale,
        requant_output_scale=qconfig5.input_scale,
        requant_out_dtype=qconfig5.input_dtype,
    )

    attn = relay.reshape(attn, [-4, -1, num_heads, -2])

    attn = relay.transpose(attn, [0, 2, 1, 3])
    req5 = relay.reshape(attn, [0, 0, -1])
    if debug_unit == f"{name}_matmul2_reshaped":
        return req5
    if debug_unit == f"{name}_pre_proj":
        return req5

    req5 = relay.reshape(req5, [-3, 0])
    proj = layers.quantized_dense(
        data=req5,
        name=name + "_attn_proj",
        input_scale=qconfig5.input_scale,
        kernel_scale=qconfig5.kernel_scale,
        units=dim,
        kernel_shape=(dim, dim),
        kernel_dtype="int8",
        add_bias=True,
    )
    qconfig6 = layers.get_qconfig(name + "_qconfig_add1")
    req6 = layers.requantize(
        proj,
        input_scale=qconfig5.output_scale,
        output_scale=qconfig6.input_scale,
        out_dtype=qconfig6.input_dtype,
    )
    proj = relay.reshape(proj, [-4, batch_size, -1, -2])
    if debug_unit == f"{name}_proj":
        return proj

    ## shortcut
    req6 = relay.reshape(req6, [-4, batch_size, -1, -2])
    if debug_unit == f"{name}_pre_add1":
        return req6

    # Residual adds should happen in float space to match the integer model's
    # dequantize-add-requantize behavior and avoid fixed-point truncation drift.
    add1 = layers.add_float(
        lhs=req6,
        rhs=shortcut,
        lhs_scale=qconfig6.input_scale,
        rhs_scale=qconfig0.input_scale,
        output_scale=qconfig6.output_scale,
    )
    if debug_unit == f"{name}_add1":
        return add1

    "MLP module"
    shortcut = add1
    ## layer_norm
    qconfig7 = layers.get_qconfig(name + "_qconfig_norm2")
    norm2_bias = relay.var(name + "_norm2_bias", shape=[dim], dtype="int64")
    norm2 = layers.quantized_layernorm(add1, norm2_bias)
    if debug_unit == f"{name}_norm2":
        return norm2

    ## dense
    qconfig8 = layers.get_qconfig(name + "_qconfig_fc1")
    req8 = layers.requantize(
        norm2,
        input_scale=qconfig7.output_scale,
        output_scale=qconfig8.input_scale,
        out_dtype=qconfig8.input_dtype,
        force_float=True,
    )

    req8 = relay.reshape(req8, [-3, 0])
    fc1 = layers.quantized_dense(
        data=req8,
        name=name + "_mlp_fc1",
        input_scale=qconfig8.input_scale,
        kernel_scale=qconfig8.kernel_scale,
        units=mlp_ratio * dim,
        kernel_shape=(mlp_ratio * dim, dim),
        kernel_dtype="int8",
        add_bias=True,
    )

    qconfig9 = layers.get_qconfig(name + "_qconfig_gelu")
    req9 = layers.requantize(
        fc1,
        input_scale=qconfig8.output_scale,
        output_scale=qconfig9.input_scale,
        out_dtype=qconfig9.input_dtype,
    )
    req9 = relay.reshape(req9, [-4, batch_size, -1, -2])

    act = layers.quantized_gelu(req9, qconfig9.input_scale)
    if debug_unit == f"{name}_post_gelu":
        return act

    qconfig10 = layers.get_qconfig(name + "_qconfig_fc2")
    req10 = layers.requantize(
        act,
        input_scale=qconfig9.output_scale,
        output_scale=qconfig10.input_scale,
        out_dtype=qconfig10.input_dtype,
    )

    req10 = relay.reshape(req10, [-3, 0])
    fc2 = layers.quantized_dense(
        data=req10,
        name=name + "_mlp_fc2",
        input_scale=qconfig10.input_scale,
        kernel_scale=qconfig10.kernel_scale,
        units=dim,
        kernel_shape=(dim, mlp_ratio * dim),
        kernel_dtype="int8",
        add_bias=True,
    )

    ## shortcut
    qconfig11 = layers.get_qconfig(name + "_qconfig_add2")
    req11 = layers.requantize(
        fc2,
        input_scale=qconfig10.output_scale,
        output_scale=qconfig11.input_scale,
        out_dtype=qconfig11.input_dtype,
    )
    req11 = relay.reshape(req11, [-4, batch_size, -1, -2])

    add2 = layers.add_float(
        lhs=req11,
        rhs=shortcut,
        lhs_scale=qconfig11.input_scale,
        rhs_scale=qconfig7.input_scale,
        output_scale=qconfig11.output_scale,
    )

    add2 = relay.annotation.stop_fusion(add2)

    return add2


def Q_VisionTransformer(
    data_shape,
    dtype="int8",
    patch_size=16,
    num_patches=196,
    in_chans=3,
    num_classes=1000,
    embed_dim=192,
    depth=12,
    num_heads=3,
    mlp_ratio=4,
    debug_unit=None,
):
    data = relay.var("data", shape=data_shape, dtype=dtype)
    if debug_unit in ("only_embed", "only_embedding"):
        qconfig_embed_conv = layers.get_qconfig("qconfig_embed_conv")
        qconfig_add = layers.get_qconfig("qconfig_addpos")
        data_nhwc = relay.layout_transform(data, src_layout="NCHW", dst_layout="NHWC")
        proj = layers.quantized_conv2d(
            data=data_nhwc,
            name="embed_conv",
            add_bias=True,
            input_channels=in_chans,
            output_channels=embed_dim,
            kernel_dtype=qconfig_embed_conv.kernel_dtype,
            input_scale=qconfig_embed_conv.input_scale,
            kernel_scale=qconfig_embed_conv.kernel_scale,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding=(0, 0),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        proj = layers.requantize(
            proj,
            input_scale=qconfig_embed_conv.output_scale,
            output_scale=qconfig_add.input_scale,
            out_dtype=qconfig_add.input_dtype,
        )
        body = relay.reshape(proj, [data_shape[0], -1, embed_dim])
        if qconfig_add.input_dtype != "int8":
            body = layers.requantize(
                body,
                input_scale=qconfig_add.input_scale,
                output_scale=qconfig_add.input_scale,
                out_dtype="int8",
            )
        return relay.Function(relay.analysis.free_vars(body), body)

    standalone_block_idx = _parse_only_deit_block(debug_unit, depth)
    if standalone_block_idx is not None:
        block_input = relay.var(
            "data",
            shape=[data_shape[0], num_patches + 1, embed_dim],
            dtype="int8",
        )
        qk_scale = (embed_dim // num_heads) ** -0.5
        body = Q_Block(
            block_input,
            name=f"block_{standalone_block_idx}",
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qk_scale=qk_scale,
            batch_size=data_shape[0],
            rounding="TONEAREST",
            debug_unit=None,
        )
        return relay.Function(relay.analysis.free_vars(body), body)

    if debug_unit in ("only_head", "only_classifier", "classifier_only"):
        body = relay.var(
            "data",
            shape=[data_shape[0], num_patches + 1, embed_dim],
            dtype="int8",
        )

        qconfig_norm = layers.get_qconfig("qconfig_norm")
        norm_bias = relay.var("norm_bias", shape=[embed_dim], dtype="int64")
        norm = layers.quantized_layernorm(body, norm_bias)

        body = relay.split(norm, 197, axis=1)
        body = relay.squeeze(body[0], axis=[1])

        qconfig_head = layers.get_qconfig("qconfig_head")
        req = layers.requantize(
            body,
            input_scale=qconfig_norm.output_scale,
            output_scale=qconfig_head.input_scale,
            out_dtype=qconfig_head.input_dtype,
            force_float=True,
        )

        head = layers.quantized_dense(
            data=req,
            name="head",
            input_scale=qconfig_head.input_scale,
            kernel_scale=qconfig_head.kernel_scale,
            units=num_classes,
            kernel_shape=(num_classes, embed_dim),
            kernel_dtype="int8",
            add_bias=True,
        )

        net = layers.dequantize(head, input_scale=qconfig_head.output_scale)
        return relay.Function(relay.analysis.free_vars(net), net)

    qconfig_embed_conv = layers.get_qconfig("qconfig_embed_conv")
    data_nhwc = relay.layout_transform(data, src_layout="NCHW", dst_layout="NHWC")
    proj = layers.quantized_conv2d(
        data=data_nhwc,
        name="embed_conv",
        add_bias=True,
        input_channels=in_chans,
        output_channels=embed_dim,
        kernel_dtype=qconfig_embed_conv.kernel_dtype,
        input_scale=qconfig_embed_conv.input_scale,
        kernel_scale=qconfig_embed_conv.kernel_scale,
        kernel_size=(patch_size, patch_size),
        strides=(patch_size, patch_size),
        padding=(0, 0),
        data_layout="NHWC",
        kernel_layout="HWIO",
    )
    qconfig_add = layers.get_qconfig("qconfig_addpos")
    proj = layers.requantize(
        proj,
        input_scale=qconfig_embed_conv.output_scale,
        output_scale=qconfig_add.input_scale,
        out_dtype=qconfig_add.input_dtype,
    )
    body = relay.reshape(proj, [data_shape[0], -1, embed_dim])
    if debug_unit == "post_patch_embed":
        return relay.Function(relay.analysis.free_vars(body), body)
    if qconfig_add.input_dtype != "int8":
        body = relay.cast(body, qconfig_add.input_dtype)

    cls_token = relay.var("cls_token_weight", shape=(1, 1, embed_dim))
    cls_token = layers.quantize(
        cls_token,
        output_scale=qconfig_add.input_scale,
        out_dtype=qconfig_add.input_dtype,
    )
    cls_tokens = relay.repeat(cls_token, data_shape[0], axis=0)

    body = relay.concatenate([cls_tokens, body], axis=1)
    if debug_unit == "post_concat":
        return relay.Function(relay.analysis.free_vars(body), body)

    pos_embed = relay.var("pos_embed_weight", shape=(1, num_patches + 1, embed_dim))
    qconfig_pos = layers.get_qconfig("qconfig_pos")
    pos_embed = layers.quantize(
        pos_embed,
        output_scale=qconfig_pos.output_scale,
        out_dtype=qconfig_add.input_dtype,
    )
    if debug_unit == "post_pos_quant":
        return relay.Function(relay.analysis.free_vars(pos_embed), pos_embed)

    # Match the ORT baseline: quantized positional add via qnn.add / QLinearAdd.
    body = layers.add(
        lhs=body,
        rhs=pos_embed,
        lhs_scale=qconfig_add.input_scale,
        rhs_scale=qconfig_pos.output_scale,
        output_scale=qconfig_add.output_scale,
    )

    body = relay.annotation.stop_fusion(body)
    if debug_unit == "post_addpos":
        return relay.Function(relay.analysis.free_vars(body), body)

    qk_scale = (embed_dim // num_heads) ** -0.5

    for i in range(depth):
        body = Q_Block(
            body,
            name="block_%d" % (i),
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qk_scale=qk_scale,
            batch_size=data_shape[0],
            rounding="TONEAREST",
            debug_unit=debug_unit,
        )
        if debug_unit is not None and debug_unit.startswith(f"block_{i}_"):
            return relay.Function(relay.analysis.free_vars(body), body)
        if debug_unit == f"post_block{i}":
            return relay.Function(relay.analysis.free_vars(body), body)

    qconfig_norm = layers.get_qconfig("qconfig_norm")
    norm_bias = relay.var("norm_bias", shape=[embed_dim], dtype="int64")
    norm = layers.quantized_layernorm(body, norm_bias)

    body = relay.split(norm, 197, axis=1)
    body = relay.squeeze(body[0], axis=[1])

    if debug_unit == "pre_head":
        return relay.Function(relay.analysis.free_vars(body), body)

    qconfig_head = layers.get_qconfig("qconfig_head")
    req = layers.requantize(
        body,
        input_scale=qconfig_norm.output_scale,
        output_scale=qconfig_head.input_scale,
        out_dtype=qconfig_head.input_dtype,
        force_float=True,
    )

    if debug_unit == "pre_head_req":
        return relay.Function(relay.analysis.free_vars(req), req)

    head = layers.quantized_dense(
        data=req,
        name="head",
        input_scale=qconfig_head.input_scale,
        kernel_scale=qconfig_head.kernel_scale,
        units=num_classes,
        kernel_shape=(num_classes, embed_dim),
        kernel_dtype="int8",
        add_bias=True,
    )

    if debug_unit == "head_int":
        return relay.Function(relay.analysis.free_vars(head), head)

    net = layers.dequantize(head, input_scale=qconfig_head.output_scale)
    if debug_unit == "head_float":
        return relay.Function(relay.analysis.free_vars(net), net)
    return relay.Function(relay.analysis.free_vars(net), net)
