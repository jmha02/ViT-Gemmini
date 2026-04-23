#!/usr/bin/env python3
"""Parse ORT raw rdcycle node logs and emit breakdown CSVs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


NODE_RE = re.compile(
    r"^\[ORT_NODE_CYCLES\],(?P<exec_plan_index>\d+),(?P<name>[^,]*),(?P<op_type>[^,]*),"
    r"(?P<provider>[^,]*),(?P<cycles>\d+)\s*$"
)


def _clean_name(name: str) -> str:
    return name.lstrip("_")


def _is_head_tail_name(name: str) -> bool:
    lower_name = _clean_name(name).lower()
    return lower_name.startswith(
        (
            "cls_tok",
            "cls_flat",
            "final_ln",
            "final_norm",
            "final_pool",
            "final_qact",
            "head",
            "logits",
        )
    )


def _iter_node_events(path: Path):
    with path.open() as f:
        for line in f:
            match = NODE_RE.match(line.strip())
            if not match:
                continue
            name = _clean_name(match.group("name"))
            yield {
                "exec_plan_index": match.group("exec_plan_index"),
                "name": name,
                "op_type": match.group("op_type"),
                "provider": match.group("provider").replace("ExecutionProvider", ""),
                "cycles": int(match.group("cycles")),
            }


def _dedup_lowered_trace_aliases(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop custom-op alias rows when ORT trace already logged the lowered node.

    In trace-enabled runs on the patched onnxruntime-riscv, the core executor emits
    one `[ORT_NODE_CYCLES]` row per executed node, while some custom ops also emit
    their own row using a friendlier `profile_label`. For decomposed RepQ graphs that
    means both of these can appear for the same Gemmini matmul:

      - `blocks_blocks_0_attn_qkv_MatMul`
      - `blocks_blocks_0_attn_qkv_MatMul_GemminiMatMulInteger`

    Keep the real lowered-node record and drop the alias so operator-level totals
    are not double-counted.
    """

    suffix_by_op = {
        "GemminiMatMulInteger": "_GemminiMatMulInteger",
        "RepQUniformMatMul": "_RepQUniformMatMul",
        "RepQLogMatMul": "_RepQLogMatMul",
    }
    names = {entry["name"] for entry in events}
    deduped: list[dict[str, Any]] = []
    dropped = 0
    for entry in events:
        suffix = suffix_by_op.get(entry["op_type"])
        if suffix is not None and f"{entry['name']}{suffix}" in names:
            dropped += 1
            continue
        deduped.append(entry)
    return deduped, dropped


def _layer_name(name: str) -> str:
    name = _clean_name(name)
    if _is_head_tail_name(name):
        return "head"
    for pattern, formatter in (
        (r"^(blk\d+)_", lambda m: m.group(1)),
        (r"^blocks_blocks_(\d+)_", lambda m: f"block_{int(m.group(1))}"),
        (r"^(layers\.\d+\.blocks\.\d+)", lambda m: m.group(1).replace(".", "_")),
        (r"^layers_(\d+)_blocks_(\d+)_", lambda m: f"stage{int(m.group(1))}_block{int(m.group(2))}"),
        (r"^(layers\.\d+\.downsample)", lambda m: m.group(1).replace(".", "_")),
        (r"^layers_(\d+)_downsample", lambda m: f"stage{int(m.group(1))}_downsample"),
        (r"^(patch_embed)", lambda m: m.group(1)),
        (r"^(patch_norm)", lambda m: m.group(1)),
        (r"^(top_qact\d+)", lambda m: m.group(1)),
        (r"^(head)", lambda m: m.group(1)),
        (r"^(pre_head)", lambda m: m.group(1)),
        (r"^(pos_add)", lambda m: m.group(1)),
        (r"^(seq_tokens)", lambda m: m.group(1)),
        (r"^(input_quant)", lambda m: m.group(1)),
    ):
        match = re.match(pattern, name)
        if match:
            return formatter(match)
    return name


def _component_name(name: str, op_type: str) -> str:
    lower_name = _clean_name(name).lower()
    lower_op = op_type.lower()

    def has_any(*parts: str) -> bool:
        return any(part in lower_name for part in parts)

    if lower_name.startswith("input_quant"):
        return "input_quant"
    if lower_name.startswith("patch_embed") or lower_name.startswith("seq_tokens"):
        return "patch_embed"
    if lower_name.startswith("pos_add"):
        return "resadd"
    if lower_name.startswith("patch_norm") or lower_name.startswith("pre_head"):
        return "layernorm"
    if lower_name.startswith(("final_ln", "final_norm")):
        return "layernorm"
    if lower_name.startswith(("final_pool", "final_qact")):
        return lower_op or "other"
    if lower_name.startswith(("cls_tok", "cls_flat", "logits")):
        return lower_op or "other"
    if "downsample" in lower_name:
        if "_norm" in lower_name:
            return "layernorm"
        if has_any("reduction", "concat", "merge"):
            return "downsample"
        return "downsample_misc"
    if any(part in lower_name for part in ("ln1", "ln2", "norm1", "norm2", "final_ln")):
        return "layernorm"
    if "qkv" in lower_name or lower_name.endswith(("_q", "_k", "_kt", "_v")):
        return "qkv"
    if has_any("attn_bias_add", "attn_mask_add"):
        return "attn_bias"
    if has_any("matmul2", "av_mm", "attn_v", "attn_perm", "attn_flat"):
        return "attn_v"
    if has_any("matmul1", "attn_mm", "attn_scores", "attn_req"):
        return "attn_scores"
    if has_any("softmax"):
        return "softmax"
    if "proj" in lower_name:
        return "proj"
    if "fc1" in lower_name:
        return "fc1"
    if "gelu" in lower_name:
        return "gelu"
    if "fc2" in lower_name:
        return "fc2"
    if has_any("res1", "res2"):
        return "resadd"
    if has_any("patch_before_norm", "top_qact") or lower_name.endswith("_req_out"):
        return "requant"
    if has_any("winpart", "winrev", "tok2img", "img2tok", "shift_", "unshift_"):
        return "window_ops"
    if lower_name.startswith("head"):
        return "head"
    if "matmul" in lower_op or "gemm" in lower_op or "_mm_" in lower_name:
        return "matmul"
    if "softmax" in lower_name or "shiftmax" in lower_op:
        return "softmax"
    if "layernorm" in lower_op or "_ln" in lower_name or "_norm" in lower_name:
        return "layernorm"
    if "gelu" in lower_name or "shiftgelu" in lower_op:
        return "gelu"
    if "qlinearadd" in lower_op or ("add" in lower_op and "res" in lower_name):
        return "resadd"
    return lower_op or "other"


def _group_key(entry: dict[str, Any], group_by: str) -> str:
    if group_by == "node":
        return entry["name"]
    if group_by == "layer":
        return _layer_name(entry["name"])
    if group_by == "component":
        return _component_name(entry["name"], entry["op_type"])
    if group_by == "layer_component":
        return f"{_layer_name(entry['name'])}:{_component_name(entry['name'], entry['op_type'])}"
    if group_by == "op_type":
        return entry["op_type"] or "unknown"
    raise ValueError(group_by)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["group", "cycles"])
        writer.writeheader()
        writer.writerows(rows)


def _summarize(events: list[dict[str, Any]], group_by: str) -> list[dict[str, Any]]:
    total_cycles = sum(entry["cycles"] for entry in events)
    grouped: dict[str, dict[str, Any]] = {}
    for entry in events:
        key = _group_key(entry, group_by)
        bucket = grouped.setdefault(
            key,
            {"group": key, "cycles": 0, "calls": 0, "providers": set(), "op_types": set()},
        )
        bucket["cycles"] += entry["cycles"]
        bucket["calls"] += 1
        if entry["provider"]:
            bucket["providers"].add(entry["provider"])
        if entry["op_type"]:
            bucket["op_types"].add(entry["op_type"])

    rows = []
    for bucket in grouped.values():
        cycles = int(bucket["cycles"])
        rows.append(
            {
                "group": bucket["group"],
                "cycles": cycles,
                "percent": 0.0 if total_cycles == 0 else (100.0 * cycles / total_cycles),
                "calls": int(bucket["calls"]),
                "providers": ",".join(sorted(bucket["providers"])),
                "op_types": ",".join(sorted(bucket["op_types"])),
            }
        )
    rows.sort(key=lambda row: row["cycles"], reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="Path to ORT stdout log with [ORT_NODE_CYCLES] lines")
    parser.add_argument("--out-prefix", type=Path, required=True, help="Output file prefix")
    args = parser.parse_args()

    events = list(_iter_node_events(args.log))
    if not events:
        raise SystemExit(f"No [ORT_NODE_CYCLES] rows found in {args.log}")
    events, dropped_alias_rows = _dedup_lowered_trace_aliases(events)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    node_rows = []
    total_cycles = sum(entry["cycles"] for entry in events)
    for entry in events:
        node_rows.append(
            {
                "name": entry["name"],
                "cycles": entry["cycles"],
                "percent": 0.0 if total_cycles == 0 else (100.0 * entry["cycles"] / total_cycles),
                "exec_plan_index": entry["exec_plan_index"],
                "provider": entry["provider"],
                "op_type": entry["op_type"],
                "layer": _layer_name(entry["name"]),
                "component": _component_name(entry["name"], entry["op_type"]),
            }
        )
    node_rows.sort(key=lambda row: row["cycles"], reverse=True)

    _write_csv(Path(f"{args.out_prefix}_node_cycles.csv"), node_rows)
    _write_csv(Path(f"{args.out_prefix}_layer_cycles.csv"), _summarize(events, "layer"))
    _write_csv(Path(f"{args.out_prefix}_component_cycles.csv"), _summarize(events, "component"))
    _write_csv(
        Path(f"{args.out_prefix}_layer_component_cycles.csv"),
        _summarize(events, "layer_component"),
    )

    summary_path = Path(f"{args.out_prefix}_summary.txt")
    with summary_path.open("w") as f:
        f.write("ORT raw rdcycle node breakdown\n")
        f.write(f"Node events: {len(events)}\n")
        f.write(f"Profiled cycles: {total_cycles}\n")
        if dropped_alias_rows:
            f.write(f"Deduped alias rows: {dropped_alias_rows}\n")
        f.write("\nTop components:\n")
        for row in _summarize(events, "component")[:12]:
            f.write(
                f"{row['group']}: {row['cycles']} cycles ({row['percent']:.3f}%), "
                f"calls={row['calls']}\n"
            )


if __name__ == "__main__":
    main()
