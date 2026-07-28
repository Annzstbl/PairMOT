#!/usr/bin/env python3
"""Measure per-draw AP variance on a repeated single-image validation set."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from evaluate_bs1_lx_common_ap import (  # noqa: E402
    evaluate, load_lx, load_mot_gt, load_ours,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-mot", required=True)
    parser.add_argument("--kind", choices=("ours", "lx"), required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--max-dets", type=int, default=100)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def statistics(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "median": float(np.median(array)),
    }


def main():
    args = parse_args()
    ground_truth, _ = load_mot_gt(args.gt_mot)
    loader = load_ours if args.kind == "ours" else load_lx
    predictions = loader(args.detections)
    image_ids = sorted(set(ground_truth) & set(predictions))
    if not image_ids:
        raise RuntimeError("no shared image IDs in GT and detections")

    per_image = {}
    for image_id in image_ids:
        one_gt = {1: ground_truth[image_id]}
        class_counts = {
            class_id: len(polygons)
            for class_id, polygons in ground_truth[image_id].items()
            if polygons
        }
        one_predictions = {1: predictions[image_id]}
        per_image[str(image_id)] = evaluate(
            one_predictions, one_gt, class_counts, args.max_dets)

    result = {
        "kind": args.kind,
        "detections": args.detections,
        "max_dets": args.max_dets,
        "images": len(image_ids),
        "AP50": statistics([
            result["AP50"] for result in per_image.values()]),
        "mAP50_95": statistics([
            result["mAP50_95"] for result in per_image.values()]),
        "per_image": per_image,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
