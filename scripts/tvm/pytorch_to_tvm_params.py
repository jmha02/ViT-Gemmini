import argparse
import os

import numpy as np
import torch

from models.ivit.layers import QConfig, QuantizeContext


DEIT_DEPTH_DEFAULT = 12
SWIN_TINY_DEPTHS = (2, 2, 6, 2)
SWIN_TINY_NUM_STAGES = len(SWIN_TINY_DEPTHS)


def clip_to_int32(arr):
    """Clip values to int32 range to prevent overflow."""
    int32_min = -(2**31)
    int32_max = 2**31 - 1
    return np.clip(arr, int32_min, int32_max).astype("int32")


DEAD_CHANNEL_THRESHOLD = 0.01


def _as_state_dict(model):
    if isinstance(model, dict):
        if "state_dict" in model and isinstance(model["state_dict"], dict):
            return model["state_dict"]
        if "model" in model and isinstance(model["model"], dict):
            return model["model"]
    return model


def _detect_model_name(model):
    keys = model.keys()
    if any(k.startswith("layers.0.blocks.0") for k in keys):
        return "swin_tiny_patch4_window7_224"
    if any(k.startswith("blocks.0") for k in keys):
        return "deit_tiny_patch16_224"
    raise RuntimeError("Unable to detect model type from checkpoint keys")


def _resolve_model_name(model, model_name=None):
    if model_name:
        return model_name
    return _detect_model_name(model)


def resolve_model_name(model, model_name=None):
    model = _as_state_dict(model)
    return _resolve_model_name(model, model_name=model_name)


def _collect_quantized_params(model):
    params = {}
    for key, tensor in model.items():
        if "weight_integer" in key:
            params[key] = tensor.cpu().numpy().astype("int8")
        if "bias_integer" in key:
            params[key] = clip_to_int32(tensor.cpu().numpy())
    return params


def _collect_scaling_factors(model):
    params = {}
    for key, tensor in model.items():
        if "scaling_factor" not in key:
            continue
        tensor_np = tensor.cpu().numpy().reshape((-1))
        if "act_scaling_factor" in key:
            params[key] = tensor_np[0]
        else:
            params[key] = tensor_np
    return params


def _softmax_scale_to_int8(scale):
    s = float(np.array(scale).reshape(-1)[0])
    return s * 256.0 if s < (1.0 / 1024.0) else s


def _fix_dead_channels(model, weight_key, sf_key, bias_key, verbose=True, label=None):
    weight = model[weight_key].cpu().numpy()
    norm_sf = model[sf_key].cpu().numpy().reshape(-1)
    bias_int = model[bias_key].cpu().numpy().reshape(-1)

    dead_mask = np.abs(weight) < DEAD_CHANNEL_THRESHOLD
    int64_info = np.iinfo(np.int64)
    bad_bias_mask = (~np.isfinite(bias_int)) | (bias_int < int64_info.min) | (bias_int > int64_info.max)
    bad_sf_mask = (~np.isfinite(norm_sf)) | (norm_sf == 0)
    fix_mask = dead_mask & (bad_bias_mask | bad_sf_mask)
    num_fixed = int(np.sum(fix_mask))

    if num_fixed > 0:
        valid_mask = ~(fix_mask)
        mean_sf = np.mean(np.abs(norm_sf[valid_mask])) if np.any(valid_mask) else 1.0

        if verbose:
            dead_indices = np.where(fix_mask)[0].tolist()
            target = label if label else bias_key
            print(f"  [FIX] {target}: {num_fixed} problematic dead channel(s) at {dead_indices}")

        bias_int[fix_mask] = 0
        norm_sf[fix_mask] = mean_sf * np.sign(norm_sf[fix_mask])
        norm_sf[fix_mask & (norm_sf == 0)] = mean_sf

    return np.rint(bias_int).astype("int64"), norm_sf, dead_mask


def fix_dead_channels_in_norm(model, block_idx, norm_name, verbose=True):
    return _fix_dead_channels(
        model,
        weight_key=f"blocks.{block_idx}.{norm_name}.weight",
        sf_key=f"blocks.{block_idx}.{norm_name}.norm_scaling_factor",
        bias_key=f"blocks.{block_idx}.{norm_name}.bias_integer",
        verbose=verbose,
        label=f"blocks.{block_idx}.{norm_name}",
    )


def fix_final_norm_dead_channels(model, verbose=True):
    return _fix_dead_channels(
        model,
        weight_key="norm.weight",
        sf_key="norm.norm_scaling_factor",
        bias_key="norm.bias_integer",
        verbose=verbose,
        label="norm (final)",
    )


def _build_deit_param_dict(model, depth):
    params = _collect_quantized_params(model)
    renamed_params = {}

    renamed_params["embed_conv_weight"] = params["patch_embed.proj.weight_integer"].transpose(
        2, 3, 1, 0
    )
    renamed_params["embed_conv_bias"] = params["patch_embed.proj.bias_integer"].reshape(-1)

    for i in range(depth):
        for key in ["weight_integer", "bias_integer"]:
            old_name = "blocks.%d.attn.qkv." % (i) + key
            new_name = "block_%d_attn_qkv_" % (i) + key[:-8]
            renamed_params[new_name] = params[old_name]

            old_name = "blocks.%d.attn.proj." % (i) + key
            new_name = "block_%d_attn_proj_" % (i) + key[:-8]
            renamed_params[new_name] = params[old_name]

            old_name = "blocks.%d.mlp.fc1." % (i) + key
            new_name = "block_%d_mlp_fc1_" % (i) + key[:-8]
            renamed_params[new_name] = params[old_name]

            old_name = "blocks.%d.mlp.fc2." % (i) + key
            new_name = "block_%d_mlp_fc2_" % (i) + key[:-8]
            renamed_params[new_name] = params[old_name]

    renamed_params["head_weight"] = params["head.weight_integer"]
    renamed_params["head_bias"] = params["head.bias_integer"]

    for i in range(depth):
        fixed_bias1, _, _ = fix_dead_channels_in_norm(model, i, "norm1", verbose=False)
        renamed_params["block_%d_norm1_bias" % i] = fixed_bias1

        fixed_bias2, _, _ = fix_dead_channels_in_norm(model, i, "norm2", verbose=False)
        renamed_params["block_%d_norm2_bias" % i] = fixed_bias2

    fixed_norm_bias, _, _ = fix_final_norm_dead_channels(model, verbose=False)
    renamed_params["norm_bias"] = fixed_norm_bias

    renamed_params["cls_token_weight"] = model["cls_token"].cpu().numpy()
    renamed_params["pos_embed_weight"] = model["pos_embed"].cpu().numpy()

    return renamed_params


def _build_swin_tiny_param_dict(model):
    params = _collect_quantized_params(model)
    renamed_params = {}

    renamed_params["patch_embed_proj_weight"] = params[
        "patch_embed.proj.weight_integer"
    ].transpose(2, 3, 1, 0)
    renamed_params["patch_embed_proj_bias"] = params[
        "patch_embed.proj.bias_integer"
    ].reshape(-1)

    patch_norm_bias, _, _ = _fix_dead_channels(
        model,
        "patch_embed.norm.weight",
        "patch_embed.norm.norm_scaling_factor",
        "patch_embed.norm.bias_integer",
        verbose=False,
    )
    renamed_params["patch_embed_norm_bias"] = patch_norm_bias

    for stage in range(SWIN_TINY_NUM_STAGES):
        depth = SWIN_TINY_DEPTHS[stage]
        for block in range(depth):
            ckpt_prefix = f"layers.{stage}.blocks.{block}"
            relay_prefix = f"stage{stage}_block{block}"

            renamed_params[f"{relay_prefix}_attn_qkv_weight"] = params[
                f"{ckpt_prefix}.attn.qkv.weight_integer"
            ]
            renamed_params[f"{relay_prefix}_attn_qkv_bias"] = params[
                f"{ckpt_prefix}.attn.qkv.bias_integer"
            ]
            renamed_params[f"{relay_prefix}_attn_proj_weight"] = params[
                f"{ckpt_prefix}.attn.proj.weight_integer"
            ]
            renamed_params[f"{relay_prefix}_attn_proj_bias"] = params[
                f"{ckpt_prefix}.attn.proj.bias_integer"
            ]
            renamed_params[f"{relay_prefix}_mlp_fc1_weight"] = params[
                f"{ckpt_prefix}.mlp.fc1.weight_integer"
            ]
            renamed_params[f"{relay_prefix}_mlp_fc1_bias"] = params[
                f"{ckpt_prefix}.mlp.fc1.bias_integer"
            ]
            renamed_params[f"{relay_prefix}_mlp_fc2_weight"] = params[
                f"{ckpt_prefix}.mlp.fc2.weight_integer"
            ]
            renamed_params[f"{relay_prefix}_mlp_fc2_bias"] = params[
                f"{ckpt_prefix}.mlp.fc2.bias_integer"
            ]

            norm1_bias, _, _ = _fix_dead_channels(
                model,
                weight_key=f"{ckpt_prefix}.norm1.weight",
                sf_key=f"{ckpt_prefix}.norm1.norm_scaling_factor",
                bias_key=f"{ckpt_prefix}.norm1.bias_integer",
                verbose=False,
            )
            renamed_params[f"{relay_prefix}_norm1_bias"] = norm1_bias

            norm2_bias, _, _ = _fix_dead_channels(
                model,
                weight_key=f"{ckpt_prefix}.norm2.weight",
                sf_key=f"{ckpt_prefix}.norm2.norm_scaling_factor",
                bias_key=f"{ckpt_prefix}.norm2.bias_integer",
                verbose=False,
            )
            renamed_params[f"{relay_prefix}_norm2_bias"] = norm2_bias

            renamed_params[f"{relay_prefix}_attn_rel_pos_bias_table_weight"] = model[
                f"{ckpt_prefix}.attn.relative_position_bias_table"
            ].cpu().numpy().astype("float32")

        if stage < SWIN_TINY_NUM_STAGES - 1:
            ckpt_prefix = f"layers.{stage}.downsample"
            relay_prefix = f"stage{stage}_downsample"

            down_norm_bias, _, _ = _fix_dead_channels(
                model,
                weight_key=f"{ckpt_prefix}.norm.weight",
                sf_key=f"{ckpt_prefix}.norm.norm_scaling_factor",
                bias_key=f"{ckpt_prefix}.norm.bias_integer",
                verbose=False,
            )
            renamed_params[f"{relay_prefix}_norm_bias"] = down_norm_bias
            renamed_params[f"{relay_prefix}_reduction_weight"] = params[
                f"{ckpt_prefix}.reduction.weight_integer"
            ]

    final_norm_bias, _, _ = _fix_dead_channels(
        model,
        weight_key="norm.weight",
        sf_key="norm.norm_scaling_factor",
        bias_key="norm.bias_integer",
        verbose=False,
    )
    renamed_params["norm_bias"] = final_norm_bias
    renamed_params["head_weight"] = params["head.weight_integer"]
    renamed_params["head_bias"] = params["head.bias_integer"]

    return renamed_params


def build_param_dict(model, depth=DEIT_DEPTH_DEFAULT, model_name=None):
    model = _as_state_dict(model)
    resolved_model = _resolve_model_name(model, model_name=model_name)

    if resolved_model.startswith("deit_"):
        return _build_deit_param_dict(model, depth=depth)
    if resolved_model == "swin_tiny_patch4_window7_224":
        return _build_swin_tiny_param_dict(model)

    raise RuntimeError(f"Unsupported model_name: {resolved_model}")


def save_params(model, depth, save_path, model_name=None):
    renamed_params = build_param_dict(model, depth=depth, model_name=model_name)
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "params.npy"), renamed_params)


def _load_qconfig_deit(model, depth):
    params = _collect_scaling_factors(model)

    QuantizeContext.qconfig_dict["qconfig_pos"] = QConfig(
        output_scale=params["qact_pos.act_scaling_factor"]
    )
    QuantizeContext.qconfig_dict["qconfig_addpos"] = QConfig(
        input_scale=params["patch_embed.qact.act_scaling_factor"],
        input_dtype="int16",
        output_scale=params["qact1.act_scaling_factor"],
    )

    conv_input_scale = params["qact_input.act_scaling_factor"]
    conv_kernel_scale = params["patch_embed.proj.conv_scaling_factor"]
    conv_output_scale = conv_input_scale * conv_kernel_scale
    QuantizeContext.qconfig_dict["qconfig_embed_conv"] = QConfig(
        input_scale=conv_input_scale,
        kernel_scale=conv_kernel_scale,
        output_scale=conv_output_scale,
    )

    for i in range(depth):
        input_scale = (
            params["qact1.act_scaling_factor"]
            if i == 0
            else params["blocks.%d.qact4.act_scaling_factor" % (i - 1)]
        )
        _, fixed_norm1_sf, _ = fix_dead_channels_in_norm(model, i, "norm1", verbose=False)
        QuantizeContext.qconfig_dict["block_%d_qconfig_norm1" % (i)] = QConfig(
            input_scale=input_scale, output_scale=fixed_norm1_sf
        )

        input_scale = params["blocks.%d.qact1.act_scaling_factor" % (i)]
        kernel_scale = params["blocks.%d.attn.qkv.fc_scaling_factor" % (i)]
        output_scale = input_scale * kernel_scale
        QuantizeContext.qconfig_dict["block_%d_qconfig_qkv" % (i)] = QConfig(
            input_scale=input_scale,
            kernel_scale=kernel_scale,
            output_scale=output_scale,
        )

        input_scale = params["blocks.%d.attn.qact1.act_scaling_factor" % (i)]
        output_scale = params["blocks.%d.attn.matmul_1.act_scaling_factor" % (i)]
        QuantizeContext.qconfig_dict["block_%d_qconfig_matmul_1" % (i)] = QConfig(
            input_scale=input_scale, output_scale=output_scale
        )

        input_scale = params["blocks.%d.attn.qact_attn1.act_scaling_factor" % (i)]
        softmax_scale = params["blocks.%d.attn.int_softmax.act_scaling_factor" % (i)]
        softmax_scale_int8 = softmax_scale * 256.0
        QuantizeContext.qconfig_dict["block_%d_qconfig_softmax" % (i)] = QConfig(
            input_scale=input_scale, output_scale=softmax_scale_int8
        )

        input_scale = softmax_scale_int8
        qact1_scale = params["blocks.%d.attn.qact1.act_scaling_factor" % (i)]
        output_scale = input_scale * qact1_scale
        QuantizeContext.qconfig_dict["block_%d_qconfig_matmul_2" % (i)] = QConfig(
            input_scale=input_scale, output_scale=output_scale
        )

        input_scale = params["blocks.%d.attn.qact2.act_scaling_factor" % (i)]
        kernel_scale = params["blocks.%d.attn.proj.fc_scaling_factor" % (i)]
        output_scale = input_scale * kernel_scale
        QuantizeContext.qconfig_dict["block_%d_qconfig_proj" % (i)] = QConfig(
            input_scale=input_scale,
            kernel_scale=kernel_scale,
            output_scale=output_scale,
        )

        input_scale = params["blocks.%d.attn.qact3.act_scaling_factor" % (i)]
        output_scale = params["blocks.%d.qact2.act_scaling_factor" % (i)]
        QuantizeContext.qconfig_dict["block_%d_qconfig_add1" % (i)] = QConfig(
            input_scale=input_scale, input_dtype="int16", output_scale=output_scale
        )

        input_scale = params["blocks.%d.qact2.act_scaling_factor" % (i)]
        _, fixed_norm2_sf, _ = fix_dead_channels_in_norm(model, i, "norm2", verbose=False)
        QuantizeContext.qconfig_dict["block_%d_qconfig_norm2" % (i)] = QConfig(
            input_scale=input_scale, output_scale=fixed_norm2_sf
        )

        input_scale = params["blocks.%d.qact3.act_scaling_factor" % (i)]
        kernel_scale = params["blocks.%d.mlp.fc1.fc_scaling_factor" % (i)]
        output_scale = input_scale * kernel_scale
        QuantizeContext.qconfig_dict["block_%d_qconfig_fc1" % (i)] = QConfig(
            input_scale=input_scale,
            kernel_scale=kernel_scale,
            output_scale=output_scale,
        )

        input_scale = params["blocks.%d.mlp.qact_gelu.act_scaling_factor" % (i)]
        output_scale = params["blocks.%d.mlp.act.act_scaling_factor" % (i)]
        QuantizeContext.qconfig_dict["block_%d_qconfig_gelu" % (i)] = QConfig(
            input_scale=input_scale, output_scale=output_scale, input_dtype="int8"
        )

        input_scale = params["blocks.%d.mlp.qact1.act_scaling_factor" % (i)]
        kernel_scale = params["blocks.%d.mlp.fc2.fc_scaling_factor" % (i)]
        output_scale = input_scale * kernel_scale
        QuantizeContext.qconfig_dict["block_%d_qconfig_fc2" % (i)] = QConfig(
            input_scale=input_scale,
            kernel_scale=kernel_scale,
            output_scale=output_scale,
        )

        input_scale = params["blocks.%d.mlp.qact2.act_scaling_factor" % (i)]
        output_scale = params["blocks.%d.qact4.act_scaling_factor" % (i)]
        QuantizeContext.qconfig_dict["block_%d_qconfig_add2" % (i)] = QConfig(
            input_scale=input_scale, input_dtype="int16", output_scale=output_scale
        )

    _, fixed_final_norm_sf, _ = fix_final_norm_dead_channels(model, verbose=False)
    last_block_output_scale = params["blocks.%d.qact4.act_scaling_factor" % (depth - 1)]
    QuantizeContext.qconfig_dict["qconfig_norm"] = QConfig(
        input_scale=last_block_output_scale, output_scale=fixed_final_norm_sf
    )

    input_scale = params["qact2.act_scaling_factor"]
    kernel_scale = params["head.fc_scaling_factor"]
    output_scale = input_scale * kernel_scale
    QuantizeContext.qconfig_dict["qconfig_head"] = QConfig(
        input_scale=input_scale, kernel_scale=kernel_scale, output_scale=output_scale
    )


def _load_qconfig_swin_tiny(model):
    params = _collect_scaling_factors(model)

    conv_input_scale = params["qact_input.act_scaling_factor"]
    conv_kernel_scale = params["patch_embed.proj.conv_scaling_factor"]
    conv_output_scale = conv_input_scale * conv_kernel_scale
    QuantizeContext.qconfig_dict["qconfig_embed_conv"] = QConfig(
        input_scale=conv_input_scale,
        kernel_scale=conv_kernel_scale,
        output_scale=conv_output_scale,
    )

    _, patch_norm_sf, _ = _fix_dead_channels(
        model,
        "patch_embed.norm.weight",
        "patch_embed.norm.norm_scaling_factor",
        "patch_embed.norm.bias_integer",
        verbose=False,
    )
    QuantizeContext.qconfig_dict["qconfig_patch_norm"] = QConfig(
        input_scale=params["patch_embed.qact_before_norm.act_scaling_factor"],
        output_scale=patch_norm_sf,
    )
    QuantizeContext.qconfig_dict["qconfig_patch_out"] = QConfig(
        input_scale=patch_norm_sf,
        input_dtype="int16",
        output_scale=params["patch_embed.qact.act_scaling_factor"],
    )
    QuantizeContext.qconfig_dict["qconfig_stem"] = QConfig(
        input_scale=params["patch_embed.qact.act_scaling_factor"],
        input_dtype="int16",
        output_scale=params["qact1.act_scaling_factor"],
    )

    current_scale = params["qact1.act_scaling_factor"]

    for stage in range(SWIN_TINY_NUM_STAGES):
        depth = SWIN_TINY_DEPTHS[stage]

        for block in range(depth):
            ckpt_prefix = f"layers.{stage}.blocks.{block}"
            relay_prefix = f"stage{stage}_block{block}"

            _, norm1_sf, _ = _fix_dead_channels(
                model,
                weight_key=f"{ckpt_prefix}.norm1.weight",
                sf_key=f"{ckpt_prefix}.norm1.norm_scaling_factor",
                bias_key=f"{ckpt_prefix}.norm1.bias_integer",
                verbose=False,
            )
            QuantizeContext.qconfig_dict[f"{relay_prefix}_qconfig_norm1"] = QConfig(
                input_scale=current_scale,
                output_scale=norm1_sf,
            )

            input_scale = params[f"{ckpt_prefix}.qact1.act_scaling_factor"]
            kernel_scale = params[f"{ckpt_prefix}.attn.qkv.fc_scaling_factor"]
            output_scale = input_scale * kernel_scale
            QuantizeContext.qconfig_dict[f"{relay_prefix}_qconfig_qkv"] = QConfig(
                input_scale=input_scale,
                kernel_scale=kernel_scale,
                output_scale=output_scale,
            )

            input_scale = params[f"{ckpt_prefix}.attn.qact1.act_scaling_factor"]
            output_scale = params[f"{ckpt_prefix}.attn.matmul_1.act_scaling_factor"]
            QuantizeContext.qconfig_dict[f"{relay_prefix}_qconfig_matmul_1"] = QConfig(
                input_scale=input_scale,
                output_scale=output_scale,
            )

            input_scale = params[f"{ckpt_prefix}.attn.qact2.act_scaling_factor"]
            softmax_scale = _softmax_scale_to_int8(
                params[f"{ckpt_prefix}.attn.log_int_softmax.act_scaling_factor"]
            )
            QuantizeContext.qconfig_dict[f"{relay_prefix}_qconfig_softmax"] = QConfig(
                input_scale=input_scale,
                output_scale=softmax_scale,
            )

            QuantizeContext.qconfig_dict[f"{relay_prefix}_qconfig_rel_pos"] = QConfig(
                output_scale=params[f"{ckpt_prefix}.attn.qact_table.act_scaling_factor"]
            )

            qact1_scale = params[f"{ckpt_prefix}.attn.qact1.act_scaling_factor"]
            output_scale = softmax_scale * qact1_scale
            QuantizeContext.qconfig_dict[f"{relay_prefix}_qconfig_matmul_2"] = QConfig(
                input_scale=softmax_scale,
                output_scale=output_scale,
            )

            input_scale = params[f"{ckpt_prefix}.attn.qact3.act_scaling_factor"]
            kernel_scale = params[f"{ckpt_prefix}.attn.proj.fc_scaling_factor"]
            output_scale = input_scale * kernel_scale
            QuantizeContext.qconfig_dict[f"{relay_prefix}_qconfig_proj"] = QConfig(
                input_scale=input_scale,
                kernel_scale=kernel_scale,
                output_scale=output_scale,
            )

            input_scale = params[f"{ckpt_prefix}.attn.qact4.act_scaling_factor"]
            output_scale = params[f"{ckpt_prefix}.qact2.act_scaling_factor"]
            QuantizeContext.qconfig_dict[f"{relay_prefix}_qconfig_add1"] = QConfig(
                input_scale=input_scale,
                input_dtype="int16",
                output_scale=output_scale,
            )

            _, norm2_sf, _ = _fix_dead_channels(
                model,
                weight_key=f"{ckpt_prefix}.norm2.weight",
                sf_key=f"{ckpt_prefix}.norm2.norm_scaling_factor",
                bias_key=f"{ckpt_prefix}.norm2.bias_integer",
                verbose=False,
            )
            QuantizeContext.qconfig_dict[f"{relay_prefix}_qconfig_norm2"] = QConfig(
                input_scale=params[f"{ckpt_prefix}.qact2.act_scaling_factor"],
                output_scale=norm2_sf,
            )

            input_scale = params[f"{ckpt_prefix}.qact3.act_scaling_factor"]
            kernel_scale = params[f"{ckpt_prefix}.mlp.fc1.fc_scaling_factor"]
            output_scale = input_scale * kernel_scale
            QuantizeContext.qconfig_dict[f"{relay_prefix}_qconfig_fc1"] = QConfig(
                input_scale=input_scale,
                kernel_scale=kernel_scale,
                output_scale=output_scale,
            )

            input_scale = params[f"{ckpt_prefix}.mlp.qact_gelu.act_scaling_factor"]
            output_scale = params[f"{ckpt_prefix}.mlp.act.act_scaling_factor"]
            QuantizeContext.qconfig_dict[f"{relay_prefix}_qconfig_gelu"] = QConfig(
                input_scale=input_scale,
                output_scale=output_scale,
                input_dtype="int8",
            )

            input_scale = params[f"{ckpt_prefix}.mlp.qact1.act_scaling_factor"]
            kernel_scale = params[f"{ckpt_prefix}.mlp.fc2.fc_scaling_factor"]
            output_scale = input_scale * kernel_scale
            QuantizeContext.qconfig_dict[f"{relay_prefix}_qconfig_fc2"] = QConfig(
                input_scale=input_scale,
                kernel_scale=kernel_scale,
                output_scale=output_scale,
            )

            input_scale = params[f"{ckpt_prefix}.mlp.qact2.act_scaling_factor"]
            output_scale = params[f"{ckpt_prefix}.qact4.act_scaling_factor"]
            QuantizeContext.qconfig_dict[f"{relay_prefix}_qconfig_add2"] = QConfig(
                input_scale=input_scale,
                input_dtype="int16",
                output_scale=output_scale,
            )

            current_scale = output_scale

        if stage < SWIN_TINY_NUM_STAGES - 1:
            ckpt_prefix = f"layers.{stage}.downsample"
            relay_prefix = f"stage{stage}_downsample"

            _, ds_norm_sf, _ = _fix_dead_channels(
                model,
                weight_key=f"{ckpt_prefix}.norm.weight",
                sf_key=f"{ckpt_prefix}.norm.norm_scaling_factor",
                bias_key=f"{ckpt_prefix}.norm.bias_integer",
                verbose=False,
            )
            QuantizeContext.qconfig_dict[f"{relay_prefix}_qconfig_norm"] = QConfig(
                input_scale=current_scale,
                output_scale=ds_norm_sf,
            )

            input_scale = params[f"{ckpt_prefix}.qact1.act_scaling_factor"]
            kernel_scale = params[f"{ckpt_prefix}.reduction.fc_scaling_factor"]
            output_scale = input_scale * kernel_scale
            QuantizeContext.qconfig_dict[f"{relay_prefix}_qconfig_reduction"] = QConfig(
                input_scale=input_scale,
                kernel_scale=kernel_scale,
                output_scale=output_scale,
            )

            downsample_out_scale = params[f"{ckpt_prefix}.qact2.act_scaling_factor"]
            QuantizeContext.qconfig_dict[f"{relay_prefix}_qconfig_out"] = QConfig(
                input_scale=output_scale,
                output_scale=downsample_out_scale,
            )

            current_scale = downsample_out_scale

    _, fixed_final_norm_sf, _ = _fix_dead_channels(
        model,
        weight_key="norm.weight",
        sf_key="norm.norm_scaling_factor",
        bias_key="norm.bias_integer",
        verbose=False,
    )
    QuantizeContext.qconfig_dict["qconfig_norm"] = QConfig(
        input_scale=current_scale,
        output_scale=fixed_final_norm_sf,
    )

    QuantizeContext.qconfig_dict["qconfig_post_norm"] = QConfig(
        input_scale=fixed_final_norm_sf,
        output_scale=params["qact2.act_scaling_factor"],
    )

    QuantizeContext.qconfig_dict["qconfig_pre_head"] = QConfig(
        input_scale=params["qact2.act_scaling_factor"],
        output_scale=params["qact3.act_scaling_factor"],
    )

    head_input_scale = params["qact3.act_scaling_factor"]
    head_kernel_scale = params["head.fc_scaling_factor"]
    head_output_scale = head_input_scale * head_kernel_scale
    QuantizeContext.qconfig_dict["qconfig_head"] = QConfig(
        input_scale=head_input_scale,
        kernel_scale=head_kernel_scale,
        output_scale=head_output_scale,
    )


def load_qconfig(model, depth=DEIT_DEPTH_DEFAULT, model_name=None):
    model = _as_state_dict(model)
    resolved_model = _resolve_model_name(model, model_name=model_name)

    QuantizeContext.qconfig_dict = {}
    QuantizeContext.qconfig_print = {}

    if resolved_model.startswith("deit_"):
        _load_qconfig_deit(model, depth=depth)
        return
    if resolved_model == "swin_tiny_patch4_window7_224":
        try:
            _load_qconfig_swin_tiny(model)
        except KeyError as exc:
            sample_keys = list(model.keys())[:12]
            raise RuntimeError(
                "Swin TVM lowering expects an I-ViT QAT checkpoint with quantization "
                f"scales such as 'qact_input.act_scaling_factor', but the provided "
                f"checkpoint is missing {exc!s}. This usually means you passed a "
                "plain FP32/pretrained Swin checkpoint instead of the quantized QAT "
                f"checkpoint. Sample keys from the provided checkpoint: {sample_keys}"
            ) from exc
        return

    raise RuntimeError(f"Unsupported model_name: {resolved_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="I-ViT convert model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="saved checkpoint path in QAT (checkpoint.pth.tar)",
    )
    parser.add_argument("--params-path", default="", help="Saved parameters directory")
    parser.add_argument("--depth", default=DEIT_DEPTH_DEFAULT, type=int, help="Depth of ViT")
    parser.add_argument(
        "--model-name",
        default=None,
        choices=["deit_tiny_patch16_224", "swin_tiny_patch4_window7_224"],
        help="Model name (omit to auto-detect from checkpoint keys)",
    )

    args = parser.parse_args()
    model = torch.load(args.model_path, map_location=torch.device("cpu"))
    save_params(model, args.depth, args.params_path, model_name=args.model_name)
