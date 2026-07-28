#!/usr/bin/env python3
"""Summarize common AP and assignment diagnostics for controlled bridges."""

import argparse
import csv
import glob
import json
from pathlib import Path
import sys


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from evaluate_bs1_lx_common_ap import (  # noqa: E402
    evaluate, load_lx, load_mot_gt, load_ours)


def parse_run(value):
    fields = value.split("::")
    if len(fields) != 4 or fields[1] not in ("ours", "lx"):
        raise argparse.ArgumentTypeError(
            "run must be NAME::ours|lx::VAL_ROOT::DIAGNOSTIC_ROOT")
    return fields


def detection_path(root, epoch, kind):
    epoch_root = Path(root) / "epoch_{:03d}".format(epoch)
    if kind == "lx":
        return epoch_root / "detections.json"
    matches = list((epoch_root / "frame_detections").glob("*.txt"))
    if len(matches) != 1:
        raise RuntimeError(
            "expected one detection txt under {}".format(epoch_root))
    return matches[0]


def assignment_summary(root, epoch):
    matches = glob.glob(str(
        Path(root) / "epoch_{:03d}_*_matches.csv".format(epoch)))
    if len(matches) != 1:
        raise RuntimeError(
            "expected one assignment CSV for epoch {} under {}".format(
                epoch, root))
    rows = list(csv.DictReader(open(matches[0], encoding="utf-8")))
    result = {}
    for layer in range(1, 7):
        selected = [row for row in rows if int(row["layer"]) == layer]
        def mean_field(field):
            return (
                sum(float(row[field]) for row in selected)
                / max(len(selected), 1))

        result[str(layer)] = {
            "positives": len(selected),
            "covered_gt": len({int(row["gt"]) for row in selected}),
            "mean_pair_iou": mean_field("match_cost_pair_iou"),
            "mean_match_l1_pair": mean_field("match_cost_l1_pair"),
            "mean_match_weighted_l1": mean_field(
                "match_cost_weighted_l1"),
            "mean_match_weighted_class": mean_field(
                "match_cost_weighted_class"),
            "mean_match_weighted_riou": mean_field(
                "match_cost_weighted_riou"),
            "mean_match_center_penalty": mean_field(
                "match_cost_center_penalty"),
            "mean_match_fg_penalty": mean_field(
                "match_cost_fg_penalty"),
            "mean_match_total": mean_field("match_cost_total"),
            "center_penalty_positives": sum(
                float(row["match_cost_center_penalty"]) > 0
                for row in selected),
            "foreground_penalty_positives": sum(
                float(row["match_cost_fg_penalty"]) > 0
                for row in selected),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-mot", required=True)
    parser.add_argument(
        "--run", action="append", type=parse_run, required=True,
        help="NAME::ours|lx::VAL_ROOT::DIAGNOSTIC_ROOT")
    parser.add_argument("--epochs", nargs="+", type=int, required=True)
    parser.add_argument("--max-dets", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ground_truth, class_counts = load_mot_gt(args.gt_mot)
    result = {
        "max_dets": args.max_dets,
        "gt_class_counts": dict(sorted(class_counts.items())),
        "epochs": {},
    }
    for epoch in args.epochs:
        epoch_result = {}
        for name, kind, val_root, diagnostic_root in args.run:
            loader = load_lx if kind == "lx" else load_ours
            epoch_result[name] = {
                "detection": evaluate(
                    loader(detection_path(val_root, epoch, kind)),
                    ground_truth, class_counts, args.max_dets),
                "assignments": assignment_summary(diagnostic_root, epoch),
            }
        result["epochs"][str(epoch)] = epoch_result

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
