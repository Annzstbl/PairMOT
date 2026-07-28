#!/usr/bin/env python3
"""Gate sequential overfit experiments using saved validation AP."""

import argparse
import json
import re
from pathlib import Path


def best_main_ap(log_path):
    pattern = re.compile(r"mAP50:95=([\d.]+), mAP50=([\d.]+)")
    values = [
        (float(match.group(1)), float(match.group(2)))
        for match in pattern.finditer(
            Path(log_path).read_text(encoding="utf-8", errors="ignore"))
    ]
    if not values:
        raise RuntimeError("no mAP records in {}".format(log_path))
    return max(value[1] for value in values)


def best_lx_ap(root):
    values = []
    for path in Path(root).glob("epoch_*/metrics.json"):
        values.append(float(json.loads(path.read_text())["mAP50"]))
    if not values:
        raise RuntimeError("no LX metrics under {}".format(root))
    return max(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-bs1-log", required=True)
    parser.add_argument("--lx-metrics-root", required=True)
    parser.add_argument("--ours-bs2-log")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = {
        "ours_bs1_best_ap50": best_main_ap(args.ours_bs1_log),
        "lx_best_ap50": best_lx_ap(args.lx_metrics_root),
    }
    result["ours_better_than_lx"] = (
        result["ours_bs1_best_ap50"] > result["lx_best_ap50"])
    if args.ours_bs2_log:
        result["ours_bs2_best_ap50"] = best_main_ap(args.ours_bs2_log)
        result["bs2_to_bs1_best_ap50_ratio"] = (
            result["ours_bs2_best_ap50"]
            / max(result["ours_bs1_best_ap50"], 1e-12))
        # "Significantly slower" is operationalized before the run: after
        # equal epochs and equal sample exposure, BS2 must retain less than
        # 80% of BS1's best AP50.
        result["bs2_significantly_slower"] = (
            result["bs2_to_bs1_best_ap50_ratio"] < 0.8)
    Path(args.output).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

    if not result["ours_better_than_lx"]:
        raise SystemExit(2)
    if args.ours_bs2_log and not result["bs2_significantly_slower"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
