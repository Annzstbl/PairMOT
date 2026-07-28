#!/usr/bin/env python3
"""Compare tensor hashes from two deterministic loader audits."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    left = json.loads(Path(args.left).read_text())
    right = json.loads(Path(args.right).read_text())
    if len(left["batches"]) != len(right["batches"]):
        raise RuntimeError("loader audits contain different batch counts")

    differences = []
    tensor_leaves = 0
    for left_batch, right_batch in zip(
            left["batches"], right["batches"]):
        if left_batch["index"] != right_batch["index"]:
            raise RuntimeError("batch indices differ")
        if left_batch["tensors"].keys() != right_batch["tensors"].keys():
            raise RuntimeError(
                f"tensor keys differ at batch {left_batch['index']}")
        for key, left_summary in left_batch["tensors"].items():
            tensor_leaves += 1
            right_summary = right_batch["tensors"][key]
            if left_summary["sha256"] != right_summary["sha256"]:
                differences.append({
                    "batch": left_batch["index"],
                    "key": key,
                    "left_sha256": left_summary["sha256"],
                    "right_sha256": right_summary["sha256"],
                })

    result = {
        "left": args.left,
        "right": args.right,
        "batch_count": len(left["batches"]),
        "tensor_leaves": tensor_leaves,
        "equal_tensor_leaves": tensor_leaves - len(differences),
        "all_tensor_leaves_bitwise_equal": not differences,
        "differences": differences,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
