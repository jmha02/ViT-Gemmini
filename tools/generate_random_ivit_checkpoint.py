#!/usr/bin/env python3
"""Generate a smoke-test I-ViT checkpoint with populated quantization buffers."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
IVIT_ROOT = REPO_ROOT / "I-ViT"

if str(IVIT_ROOT) not in sys.path:
    sys.path.insert(0, str(IVIT_ROOT))

from models.model_utils import freeze_model  # noqa: E402
from models.swin_quant import swin_tiny_patch4_window7_224  # noqa: E402
from models.vit_quant import deit_tiny_patch16_224  # noqa: E402


BUILDERS = {
    "deit_tiny_patch16_224": deit_tiny_patch16_224,
    "swin_tiny_patch4_window7_224": swin_tiny_patch4_window7_224,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", choices=sorted(BUILDERS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-passes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    model = BUILDERS[args.model_name](pretrained=False)
    model.eval()

    with torch.no_grad():
        for _ in range(max(1, args.num_passes)):
            sample = torch.randn(args.batch_size, 3, 224, 224)
            _ = model(sample)

        freeze_model(model)
        final_sample = torch.randn(1, 3, 224, 224)
        _ = model(final_sample)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, args.output)
    print(f"Saved random I-ViT checkpoint: {args.output}")


if __name__ == "__main__":
    main()
