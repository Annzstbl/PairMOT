#!/usr/bin/env python3
"""Compare model/RNG summaries emitted by controlled model verification."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def tensor_group(left, right, group):
    left_hashes = left[group]["tensor_sha256"]
    right_hashes = right[group]["tensor_sha256"]
    common = set(left_hashes) & set(right_hashes)
    equal = sorted(
        name for name in common
        if left_hashes[name] == right_hashes[name])
    different = sorted(common - set(equal))
    return {
        "left_tensors": len(left_hashes),
        "right_tensors": len(right_hashes),
        "common_tensors": len(common),
        "equal_common_tensors": len(equal),
        "different_common_tensors": different,
        "left_only_tensors": sorted(set(left_hashes) - set(right_hashes)),
        "right_only_tensors": sorted(set(right_hashes) - set(left_hashes)),
    }


def main():
    args = parse_args()
    left = json.loads(Path(args.left).read_text())
    right = json.loads(Path(args.right).read_text())
    result = {
        "left": args.left,
        "right": args.right,
        "parameters": tensor_group(left, right, "parameters"),
        "state": tensor_group(left, right, "state"),
        "post_build_rng": {
            key: {
                "left": left["post_build_rng"][key],
                "right": right["post_build_rng"][key],
                "equal": (
                    left["post_build_rng"][key]
                    == right["post_build_rng"][key]),
            }
            for key in sorted(left["post_build_rng"])
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
