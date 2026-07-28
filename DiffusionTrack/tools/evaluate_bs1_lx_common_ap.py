#!/usr/bin/env python3
"""Evaluate our and LX overfit caches with one polygon-IoU AP protocol.

HSMOT MOT rows are:
  frame, track_id, qbox8, ignored, class_id, truncated
The class is column 11 (zero-based); column 12 is metadata and must never be
used as a class label.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
import numpy as np


IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)


def polygon_iou(first, second):
    first = np.asarray(first, dtype=np.float32).reshape(4, 2)
    second = np.asarray(second, dtype=np.float32).reshape(4, 2)
    first_area = abs(cv2.contourArea(first))
    second_area = abs(cv2.contourArea(second))
    intersection = cv2.intersectConvexConvex(first, second)[0]
    return float(intersection / max(
        first_area + second_area - intersection, 1e-12))


def interpolated_ap(tp, fp, num_gt):
    tp = np.cumsum(np.asarray(tp, dtype=np.float64))
    fp = np.cumsum(np.asarray(fp, dtype=np.float64))
    recall = tp / num_gt
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(
        recall, np.linspace(0.0, 1.0, 101), side="left")
    sampled = np.zeros(101, dtype=np.float64)
    valid = indices < len(precision)
    sampled[valid] = precision[indices[valid]]
    return float(sampled.mean())


def load_mot_gt(path):
    ground_truth = defaultdict(lambda: defaultdict(list))
    class_counts = Counter()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        values = line.split(",")
        if len(values) < 13:
            raise ValueError("expected 13-column HSMOT row: {}".format(line))
        image_id = int(float(values[0]))
        class_id = int(float(values[11]))
        truncated = int(float(values[12]))
        if truncated not in (0, 1):
            raise ValueError(
                "column 12 is expected to be the truncation flag")
        polygon = np.asarray(values[2:10], dtype=np.float32).reshape(4, 2)
        ground_truth[image_id][class_id].append(polygon)
        class_counts[class_id] += 1
    return ground_truth, class_counts


def load_ours(path):
    predictions = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        values = line.split(",")
        predictions[int(values[0])].append((
            float(values[11]),
            int(values[10]),
            np.asarray(values[2:10], dtype=np.float32).reshape(4, 2),
        ))
    return predictions


def load_lx(path):
    predictions = defaultdict(list)
    for detection in json.loads(Path(path).read_text()):
        cx, cy, width, height, angle = detection["rbox"]
        polygon = cv2.boxPoints(
            ((float(cx), float(cy)),
             (float(width), float(height)), float(angle)))
        # The isolated COCO annotations use contiguous 1-based category IDs
        # generated directly from HSMOT's contiguous 0-based labels.
        predictions[int(detection["image_id"])].append((
            float(detection["score"]),
            int(detection["category_id"]) - 1,
            polygon,
        ))
    return predictions


def evaluate(predictions, ground_truth, class_counts, max_dets):
    class_results = {}
    for class_id, num_gt in sorted(class_counts.items()):
        records = []
        for image_id, image_gt in ground_truth.items():
            gt_polygons = image_gt[class_id]
            selected = sorted(
                predictions[image_id], key=lambda item: item[0],
                reverse=True)[:max_dets]
            for score, predicted_class, polygon in selected:
                if predicted_class != class_id:
                    continue
                overlaps = np.asarray(
                    [polygon_iou(polygon, gt) for gt in gt_polygons],
                    dtype=np.float32)
                records.append((score, image_id, overlaps))
        records.sort(key=lambda item: item[0], reverse=True)

        aps = []
        for threshold in IOU_THRESHOLDS:
            used = {}
            tp, fp = [], []
            for _, image_id, overlaps in records:
                matched = used.setdefault(
                    image_id, np.zeros(len(overlaps), dtype=bool))
                candidates = np.where(~matched)[0]
                positive = False
                if len(candidates):
                    best = candidates[np.argmax(overlaps[candidates])]
                    if overlaps[best] >= threshold:
                        matched[best] = True
                        positive = True
                tp.append(float(positive))
                fp.append(float(not positive))
            aps.append(interpolated_ap(tp, fp, num_gt))
        class_results[class_id] = aps

    values = np.asarray(list(class_results.values()), dtype=np.float64)
    return {
        "AP50": float(values[:, 0].mean()),
        "mAP50_95": float(values.mean()),
        "per_class": {
            str(class_id): {
                "num_gt": int(class_counts[class_id]),
                "AP50": float(aps[0]),
                "mAP50_95": float(np.mean(aps)),
            }
            for class_id, aps in class_results.items()
        },
    }


def cache_path(root, epoch, kind):
    epoch_root = Path(root) / "epoch_{:03d}".format(epoch)
    if kind == "ours":
        matches = list((epoch_root / "frame_detections").glob("*.txt"))
        if len(matches) != 1:
            raise RuntimeError(
                "expected one frame detection file under {}".format(
                    epoch_root))
        return matches[0]
    return epoch_root / "detections.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-root", required=True)
    parser.add_argument("--lx-root", required=True)
    parser.add_argument("--gt-mot", required=True)
    parser.add_argument("--epochs", nargs="+", type=int, required=True)
    parser.add_argument("--max-dets", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()

    ground_truth, class_counts = load_mot_gt(args.gt_mot)
    result = {
        "protocol": {
            "iou": "convex polygon IoU",
            "thresholds": IOU_THRESHOLDS.tolist(),
            "ap": "101-point interpolated",
            "max_dets_per_image": args.max_dets,
            "gt_class_column_zero_based": 11,
            "gt_truncation_column_zero_based": 12,
            "lx_category_to_hsmot_class": "category_id - 1",
        },
        "gt_class_counts": dict(sorted(class_counts.items())),
        "epochs": {},
    }
    for epoch in args.epochs:
        result["epochs"][str(epoch)] = {
            "ours": evaluate(
                load_ours(cache_path(args.ours_root, epoch, "ours")),
                ground_truth, class_counts, args.max_dets),
            "lx": evaluate(
                load_lx(cache_path(args.lx_root, epoch, "lx")),
                ground_truth, class_counts, args.max_dets),
        }

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
