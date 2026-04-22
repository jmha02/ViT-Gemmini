#!/usr/bin/env python3
"""Summarize ORT Spike logs and check Gemmini usage evidence."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    calls = len(re.findall(r"Called into systolic matmul!", text))
    cycles_match = re.findall(r"Done!\s*Inference took\s*([0-9]+)\s*cycles", text)
    cycles = int(cycles_match[-1]) if cycles_match else None
    gemmini_cfg = "Gemmini extension configured with:" in text
    mode_match = re.search(r"Mode \(-x\)\s*:\s*.*\((\d)\)", text)
    mode = int(mode_match.group(1)) if mode_match else None
    return {
        "path": path,
        "calls": calls,
        "cycles": cycles,
        "gemmini_cfg": gemmini_cfg,
        "mode": mode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify ORT Spike Gemmini usage from log files"
    )
    parser.add_argument("--x1-log", type=Path, required=True, help="Spike log from -x 1 run")
    parser.add_argument("--x0-log", type=Path, default=None, help="Optional Spike log from -x 0 run")
    args = parser.parse_args()

    x1 = parse_log(args.x1_log)
    print(f"[x1] {x1['path']}")
    print(f"  systolic matmul calls : {x1['calls']}")
    print(f"  cycles                : {x1['cycles']}")
    print(f"  Gemmini configured    : {x1['gemmini_cfg']}")

    ok = True
    if x1["calls"] <= 0:
        print("  [FAIL] No systolic matmul call found in x1 log.")
        ok = False
    if not x1["gemmini_cfg"]:
        print("  [FAIL] Gemmini extension configuration print not found in x1 log.")
        ok = False

    if args.x0_log is not None:
        x0 = parse_log(args.x0_log)
        print(f"\n[x0] {x0['path']}")
        print(f"  systolic matmul calls : {x0['calls']}")
        print(f"  cycles                : {x0['cycles']}")
        print(f"  Gemmini configured    : {x0['gemmini_cfg']}")

        if x0["cycles"] is not None and x1["cycles"] is not None and x1["cycles"] > 0:
            speedup = x0["cycles"] / x1["cycles"]
            print(f"\nSpeedup (x0/x1): {speedup:.4f}x")
            if speedup <= 1.0:
                print("  [WARN] x1 is not faster than x0.")

    if ok:
        print("\n[PASS] Gemmini usage evidence found for ONNX-RT x1 run.")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
