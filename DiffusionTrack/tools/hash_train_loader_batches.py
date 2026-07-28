#!/usr/bin/env python3
"""Hash deterministic train-loader batches across current and LX environments."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolox.exp import get_exp  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dump-first-npz",
        default=None,
        help="Optionally dump every tensor leaf from the first batch for exact cross-run comparison.",
    )
    parser.add_argument("--seed", type=int, default=8823)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=20)
    return parser.parse_args()


def tensor_summary(tensor):
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    numeric = value.float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "min": float(numeric.min()) if numeric.numel() else None,
        "max": float(numeric.max()) if numeric.numel() else None,
        "mean": float(numeric.mean()) if numeric.numel() else None,
    }


def flatten_tensors(value, prefix="batch"):
    if torch.is_tensor(value):
        return {prefix: tensor_summary(value)}
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(flatten_tensors(item, f"{prefix}.{index}"))
        return result
    if isinstance(value, dict):
        result = {}
        for key, item in sorted(value.items()):
            result.update(flatten_tensors(item, f"{prefix}.{key}"))
        return result
    return {}


def flatten_strings(value, prefix="batch"):
    if isinstance(value, str):
        return {prefix: value}
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(flatten_strings(item, f"{prefix}.{index}"))
        return result
    if isinstance(value, dict):
        result = {}
        for key, item in sorted(value.items()):
            result.update(flatten_strings(item, f"{prefix}.{key}"))
        return result
    return {}


def flatten_tensor_values(value, prefix="batch"):
    if torch.is_tensor(value):
        return {prefix: value.detach().cpu().contiguous().numpy()}
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(flatten_tensor_values(item, f"{prefix}.{index}"))
        return result
    if isinstance(value, dict):
        result = {}
        for key, item in sorted(value.items()):
            result.update(flatten_tensor_values(item, f"{prefix}.{key}"))
        return result
    return {}


def main():
    args = parse_args()
    exp = get_exp(args.config, None)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    loader = exp.get_data_loader(args.batch_size, is_distributed=False)
    iterator = iter(loader)
    batches = []
    for index in range(args.num_batches):
        batch = next(iterator)
        if index == 0 and args.dump_first_npz:
            dump = Path(args.dump_first_npz)
            dump.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dump, **flatten_tensor_values(batch))
        batches.append({
            "index": index,
            "tensors": flatten_tensors(batch),
            # Paths are useful provenance but are intentionally excluded from
            # tensor equality: current and LX use different dataset roots.
            "strings": flatten_strings(batch),
        })
    result = {
        "config": args.config,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "batches": batches,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
