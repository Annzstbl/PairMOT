#!/usr/bin/env python3
"""Compare two first-step forward/backward diagnostics."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def canonical_loss_name(name):
    return name.replace("loss_giou", "loss_iou")


def main():
    args = parse_args()
    left = json.loads(Path(args.left).read_text())
    right = json.loads(Path(args.right).read_text())

    left_losses = {
        canonical_loss_name(key): value
        for key, value in left["losses"].items()
    }
    right_losses = {
        canonical_loss_name(key): value
        for key, value in right["losses"].items()
    }
    losses = {}
    for key in sorted(set(left_losses) | set(right_losses)):
        lhs = left_losses.get(key)
        rhs = right_losses.get(key)
        losses[key] = {
            "left": lhs,
            "right": rhs,
            "absolute_difference": (
                abs(lhs - rhs) if lhs is not None and rhs is not None
                else None),
            "relative_difference": (
                abs(lhs - rhs) / max(abs(rhs), 1e-30)
                if lhs is not None and rhs is not None else None),
        }

    batch_keys = set(left["batch"]) | set(right["batch"])
    batch = {
        key: {
            "left_sha256": left["batch"].get(key, {}).get("sha256"),
            "right_sha256": right["batch"].get(key, {}).get("sha256"),
            "equal": (
                left["batch"].get(key, {}).get("sha256")
                == right["batch"].get(key, {}).get("sha256")),
        }
        for key in sorted(batch_keys)
    }

    gradient_groups = {}
    for key in sorted(
            set(left["gradients"]["groups"])
            | set(right["gradients"]["groups"])):
        lhs = left["gradients"]["groups"].get(key)
        rhs = right["gradients"]["groups"].get(key)
        gradient_groups[key] = {
            "left": lhs,
            "right": rhs,
            "l2_relative_difference": (
                abs(lhs["l2"] - rhs["l2"]) / max(abs(rhs["l2"]), 1e-30)
                if lhs is not None and rhs is not None else None),
        }

    left_selected = left["gradients"]["selected"]
    right_selected = right["gradients"]["selected"]
    selected = {
        key: {
            "left": left_selected.get(key),
            "right": right_selected.get(key),
            "sha256_equal": (
                left_selected.get(key, {}).get("sha256")
                == right_selected.get(key, {}).get("sha256")),
        }
        for key in sorted(set(left_selected) | set(right_selected))
    }

    rng = {}
    for stage in sorted(set(left["rng"]) | set(right["rng"])):
        rng[stage] = {}
        for source in sorted(
                set(left["rng"].get(stage, {}))
                | set(right["rng"].get(stage, {}))):
            lhs = left["rng"].get(stage, {}).get(source)
            rhs = right["rng"].get(stage, {}).get(source)
            rng[stage][source] = {
                "left": lhs,
                "right": rhs,
                "equal": lhs == rhs,
            }

    result = {
        "left": args.left,
        "right": args.right,
        "all_batch_tensors_equal": all(
            item["equal"] for item in batch.values()),
        "batch": batch,
        "losses": losses,
        "gradient_groups": gradient_groups,
        "selected_gradients": selected,
        "rng": rng,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
