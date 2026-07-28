#!/usr/bin/env python3
"""Compare captured per-layer logits and rotated boxes from two first steps."""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--left-box-angle", choices=("native", "radians"), default="native")
    parser.add_argument(
        "--right-box-angle", choices=("native", "radians"), default="native")
    return parser.parse_args()


def tensor_comparison(left, right):
    left = left.float()
    right = right.float()
    difference = left - right
    denominator = right.norm().clamp_min(torch.finfo(torch.float32).eps)
    return {
        "shape": list(left.shape),
        "exact": bool(torch.equal(left, right)),
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "rmse": float(difference.square().mean().sqrt()),
        "relative_l2": float(difference.norm() / denominator),
    }


def main():
    args = parse_args()
    left = torch.load(args.left, map_location="cpu", weights_only=True)
    right = torch.load(args.right, map_location="cpu", weights_only=True)
    result = {}
    for layer in sorted(set(left) | set(right)):
        result[layer] = {}
        for name in sorted(set(left.get(layer, {})) | set(right.get(layer, {}))):
            if name not in left.get(layer, {}) or name not in right.get(layer, {}):
                result[layer][name] = {"missing": True}
            else:
                left_value = left[layer][name]
                right_value = right[layer][name]
                if name.endswith("_boxes"):
                    if args.left_box_angle == "radians":
                        left_value = left_value.clone()
                        left_value[..., 4] = torch.rad2deg(
                            left_value[..., 4])
                    if args.right_box_angle == "radians":
                        right_value = right_value.clone()
                        right_value[..., 4] = torch.rad2deg(
                            right_value[..., 4])
                result[layer][name] = tensor_comparison(
                    left_value, right_value)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
