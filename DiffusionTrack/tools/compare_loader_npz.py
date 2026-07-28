#!/usr/bin/env python3
"""Compare two first-batch NPZ dumps exactly and numerically."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-differences", type=int, default=100)
    return parser.parse_args()


def digest(array):
    return hashlib.sha256(
        np.ascontiguousarray(array).view(np.uint8).tobytes()).hexdigest()


def main():
    args = parse_args()
    left = np.load(args.left)
    right = np.load(args.right)
    if left.files != right.files:
        raise RuntimeError(
            f"NPZ key order differs: {left.files!r} != {right.files!r}")

    arrays = {}
    all_bitwise_equal = True
    for key in left.files:
        lhs = left[key]
        rhs = right[key]
        if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
            arrays[key] = {
                "left_shape": list(lhs.shape),
                "right_shape": list(rhs.shape),
                "left_dtype": str(lhs.dtype),
                "right_dtype": str(rhs.dtype),
                "compatible": False,
            }
            all_bitwise_equal = False
            continue

        bitwise_equal = digest(lhs) == digest(rhs)
        all_bitwise_equal &= bitwise_equal
        lhs_bytes = np.ascontiguousarray(lhs).view(np.uint8)
        rhs_bytes = np.ascontiguousarray(rhs).view(np.uint8)
        element_mask = lhs != rhs
        difference_indices = np.argwhere(element_mask)
        numeric = lhs.dtype.kind in "fiu"
        details = []
        for index in difference_indices[:args.max_differences]:
            index_tuple = tuple(int(component) for component in index)
            details.append({
                "index": list(index_tuple),
                "left": float(lhs[index_tuple]),
                "right": float(rhs[index_tuple]),
                "delta": float(
                    lhs[index_tuple].astype(np.float64)
                    - rhs[index_tuple].astype(np.float64)),
            })
        arrays[key] = {
            "compatible": True,
            "shape": list(lhs.shape),
            "dtype": str(lhs.dtype),
            "bitwise_equal": bitwise_equal,
            "left_sha256": digest(lhs),
            "right_sha256": digest(rhs),
            "different_bytes": int(np.count_nonzero(lhs_bytes != rhs_bytes)),
            "different_elements": int(len(difference_indices)),
            "max_absolute_difference": (
                float(np.max(np.abs(
                    lhs.astype(np.float64) - rhs.astype(np.float64))))
                if numeric and lhs.size else None),
            "differences": details,
        }

    result = {
        "left": args.left,
        "right": args.right,
        "all_bitwise_equal": all_bitwise_equal,
        "arrays": arrays,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
