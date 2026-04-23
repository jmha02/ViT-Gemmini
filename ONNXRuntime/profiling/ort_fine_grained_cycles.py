#!/usr/bin/env python3
"""Build finer ORT cycle breakdowns from exact node-cycle CSVs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


GEMM_OP_TYPES = {
    "GemminiMatMulInteger",
    "RepQUniformMatMul",
    "RepQLogMatMul",
    "QLinearConv",
}

REQUANT_OP_PREFIXES = ("Requantize",)
LAYOUT_OP_TYPES = {"Reshape", "Transpose", "Gather", "Split", "Concat", "Slice"}


def _clean_name(name: str) -> str:
    return name.lstrip("_")


def _normalize_layer(name: str) -> str:
    name = _clean_name(name)
    lower_name = name.lower()
    if lower_name == "input_quant":
        return "input_quant"
    if lower_name.startswith(
        (
            "patch_embed",
            "patch_norm",
            "patch_before_norm",
            "top_qact",
            "seq_tokens",
            "pos_add",
        )
    ):
        return "patch_embed"
    match = re.match(r"^blk(\d+)_", name)
    if match:
        return f"block_{int(match.group(1))}"
    match = re.match(r"^blocks_blocks_(\d+)_", name)
    if match:
        return f"block_{int(match.group(1))}"
    match = re.match(r"^layers\.(\d+)\.blocks\.(\d+)", name)
    if match:
        return f"stage{int(match.group(1))}_block{int(match.group(2))}"
    match = re.match(r"^layers_(\d+)_blocks_(\d+)_", name)
    if match:
        return f"stage{int(match.group(1))}_block{int(match.group(2))}"
    match = re.match(r"^layers\.(\d+)\.downsample", name)
    if match:
        return f"stage{int(match.group(1))}_downsample"
    match = re.match(r"^layers_(\d+)_downsample", name)
    if match:
        return f"stage{int(match.group(1))}_downsample"
    if lower_name.startswith(
        (
            "cls_tok",
            "cls_flat",
            "final_ln",
            "final_norm",
            "final_pool",
            "final_qact",
            "final_flat",
            "head",
            "logits",
        )
    ):
        return "head"
    return name.replace(".", "_")


def _is_block_layer(layer: str) -> bool:
    return layer.startswith("block_") or bool(re.fullmatch(r"stage\d+_block\d+", layer))


def _is_downsample_layer(layer: str) -> bool:
    return bool(re.fullmatch(r"stage\d+_downsample", layer))


def _is_gemm_op(op_type: str) -> bool:
    return op_type in GEMM_OP_TYPES or "matmul" in op_type.lower() or "gemm" in op_type.lower()


def _is_requant_op(op_type: str) -> bool:
    return op_type.startswith(REQUANT_OP_PREFIXES)


def _backend(micro_component: str) -> str:
    if micro_component.endswith("_gemm") or micro_component in {"patch_embed_gemm", "head_gemm"}:
        return "gemmini"
    return "cpu"


def _classify_residual(prefix: str, name: str, op_type: str) -> str:
    if op_type in {"QLinearAdd", "Add"} or name.endswith("_sum_f32"):
        return f"{prefix}_add"
    if any(token in name for token in ("cast_f32", "dequant")):
        return f"{prefix}_dequant"
    if (
        _is_requant_op(op_type)
        or op_type in {"QuantizeLinear", "Div", "Round", "Clip", "Cast"}
        or any(token in name for token in ("_q_", "_scaled", "_rounded", "_clipped"))
    ):
        return f"{prefix}_quant"
    return f"{prefix}_add"


def _classify_patch(name: str, op_type: str) -> str:
    if name.startswith("patch_before_norm"):
        return "patch_pre"
    if name.startswith("patch_norm"):
        return "patch_norm_requant" if _is_requant_op(op_type) else "patch_norm"
    if name.startswith("top_qact"):
        return "patch_post"
    if name.startswith("pos_add"):
        return "patch_post"
    if name.startswith(("patch_embed", "seq_tokens")):
        if _is_gemm_op(op_type):
            return "patch_embed_gemm"
        if "biased" in name:
            return "patch_embed_bias"
        if _is_requant_op(op_type):
            return "patch_embed_requant"
        if op_type in LAYOUT_OP_TYPES or any(token in name for token in ("split", "reorder", "tokens")):
            return "patch_embed_layout"
        return "patch_embed"
    return "misc"


def _classify_window(name: str) -> str | None:
    if "tok2img" in name:
        return "tok2img"
    if "img2tok" in name:
        return "img2tok"
    if "winpart" in name:
        return "window_partition"
    if "winrev" in name:
        return "window_reverse"
    if "unshift_" in name:
        return "unshift"
    if "shift_" in name:
        return "shift"
    return None


def _classify_block(name: str, op_type: str) -> str:
    window_component = _classify_window(name)
    if window_component is not None:
        return window_component

    if any(token in name for token in ("_ln1_", "_norm1_")) or name.endswith(("_ln1_out", "_norm1_out")):
        return "norm1_requant" if _is_requant_op(op_type) else "norm1"
    if any(token in name for token in ("_ln2_", "_norm2_")) or name.endswith(("_ln2_out", "_norm2_out")):
        return "norm2_requant" if _is_requant_op(op_type) else "norm2"

    if "attn_mask_add" in name:
        return "attn_mask"
    if "attn_bias_add" in name:
        return "attn_bias"
    if "softmax" in name or op_type == "Shiftmax":
        return "softmax"

    if "attn_v_req" in name:
        return "attn_v_requant"
    if any(token in name for token in ("attn_perm", "attn_flat")):
        return "attn_v_layout"
    if _is_gemm_op(op_type) and any(token in name for token in ("matmul2", "av_mm", "attn_v")):
        return "attn_v_gemm"

    if "attn_req" in name:
        return "attn_scores_requant"
    if _is_gemm_op(op_type) and any(token in name for token in ("matmul1", "attn_mm", "attn_scores")):
        return "attn_scores_gemm"

    if any(token in name for token in ("qkv_5d", "qkv_transposed")) or re.search(r"_(q|k|kt|v)$", name):
        return "qkv_layout"
    if "qkv_req" in name:
        return "qkv_requant"
    if "qkv_biased" in name:
        return "qkv_bias"
    if _is_gemm_op(op_type) and ("qkv" in name):
        return "qkv_gemm"

    if "proj_req" in name:
        return "proj_requant"
    if "proj_biased" in name:
        return "proj_bias"
    if _is_gemm_op(op_type) and "proj" in name:
        return "proj_gemm"

    if "res1" in name:
        return _classify_residual("res1", name, op_type)
    if "res2" in name:
        return _classify_residual("res2", name, op_type)

    if "_mlp_fc1_" in name or "_fc1_" in name:
        if _is_gemm_op(op_type):
            return "fc1_gemm"
        if "biased" in name:
            return "fc1_bias"
        if _is_requant_op(op_type):
            return "fc1_requant"
    if "_mlp_gelu_" in name or "_gelu_" in name or op_type == "ShiftGELU":
        return "gelu_requant" if _is_requant_op(op_type) else "gelu"
    if "_mlp_fc2_" in name or "_fc2_" in name:
        if _is_gemm_op(op_type):
            return "fc2_gemm"
        if "biased" in name:
            return "fc2_bias"
        if _is_requant_op(op_type):
            return "fc2_requant"

    if op_type in LAYOUT_OP_TYPES:
        return "layout"
    if _is_requant_op(op_type):
        return "misc_requant"
    return "misc"


def _classify_downsample(name: str, op_type: str) -> str:
    if "_norm_req" in name:
        return "downsample_norm_requant"
    if "_norm_out" in name:
        return "downsample_norm"
    if "_reduction_req" in name:
        return "downsample_reduction_requant"
    if "_reduction_biased" in name:
        return "downsample_reduction_bias"
    if "_reduction_mm_" in f"_{name}_" or (_is_gemm_op(op_type) and "reduction" in name):
        return "downsample_reduction_gemm"
    if any(token in name for token in ("tok2img", "img2tok", "_x0_", "_x1_", "_x2_", "_x3_", "_merged")):
        return "downsample_merge"
    return "misc"


def _classify_head(name: str, op_type: str) -> str:
    if name.startswith(("cls_tok", "cls_flat")):
        return "head_token"
    if name.startswith(("final_ln", "final_norm")):
        return "head_norm_requant" if _is_requant_op(op_type) else "head_norm"
    if name.startswith(("final_pool", "final_qact", "final_flat")):
        return "head_pool"
    if name.startswith("head_"):
        if _is_gemm_op(op_type):
            return "head_gemm"
        if "biased" in name:
            return "head_bias"
        if _is_requant_op(op_type):
            return "head_requant"
        return "head_misc"
    if name.startswith("logits"):
        return "head_logits"
    return "head_misc"


def _micro_component(name: str, op_type: str) -> str:
    name = _clean_name(name)
    layer = _normalize_layer(name)
    if layer == "input_quant":
        return "input_quant"
    if layer == "patch_embed":
        return _classify_patch(name, op_type)
    if _is_block_layer(layer):
        return _classify_block(name, op_type)
    if _is_downsample_layer(layer):
        return _classify_downsample(name, op_type)
    if layer == "head":
        return _classify_head(name, op_type)
    return "misc"


def _summarize(rows: list[dict[str, Any]], group_field: str) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = defaultdict(lambda: {"cycles": 0, "calls": 0, "providers": set(), "op_types": set()})
    total_cycles = sum(int(row["cycles"]) for row in rows)
    for row in rows:
        group = row[group_field]
        totals[group]["cycles"] += int(row["cycles"])
        totals[group]["calls"] += 1
        if row["provider"]:
            totals[group]["providers"].add(row["provider"])
        if row["op_type"]:
            totals[group]["op_types"].add(row["op_type"])

    out = []
    for group, info in totals.items():
        cycles = int(info["cycles"])
        out.append(
            {
                "group": group,
                "cycles": cycles,
                "percent": 0.0 if total_cycles == 0 else (100.0 * cycles / total_cycles),
                "calls": int(info["calls"]),
                "providers": ",".join(sorted(info["providers"])),
                "op_types": ",".join(sorted(info["op_types"])),
                "backend": _backend(group),
            }
        )
    out.sort(key=lambda row: row["cycles"], reverse=True)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else ["group", "cycles"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-csv", type=Path, required=True, help="Node cycle CSV from ort_log_node_cycles.py")
    parser.add_argument("--out-prefix", type=Path, required=True, help="Output file prefix")
    args = parser.parse_args()

    with args.node_csv.open() as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"No rows found in {args.node_csv}")

    enriched = []
    for row in rows:
        name = _clean_name(row["name"])
        op_type = row["op_type"]
        layer = _normalize_layer(name)
        micro_component = _micro_component(name, op_type)
        enriched.append(
            {
                **row,
                "name": name,
                "layer": layer,
                "micro_component": micro_component,
                "backend": _backend(micro_component),
            }
        )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(Path(f"{args.out_prefix}_node_cycles_fine.csv"), enriched)
    _write_csv(Path(f"{args.out_prefix}_micro_component_cycles.csv"), _summarize(enriched, "micro_component"))
    _write_csv(
        Path(f"{args.out_prefix}_layer_micro_component_cycles.csv"),
        _summarize(
            [
                {**row, "layer_micro_component": f"{row['layer']}:{row['micro_component']}"}
                for row in enriched
            ],
            "layer_micro_component",
        ),
    )


if __name__ == "__main__":
    main()
