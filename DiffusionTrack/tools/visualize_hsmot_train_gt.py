#!/usr/bin/env python3
"""Render post-augmentation HSMOT pair samples with their training GT."""

import argparse
import json
import os
import random

import cv2
import numpy as np
import torch

from yolox.exp import get_exp


CLASS_NAMES = (
    "car", "bike", "pedestrian", "van", "truck", "bus",
    "tricycle", "awning-bike",
)
COLORS = (
    (40, 220, 40), (255, 170, 30), (40, 220, 255), (230, 60, 230),
    (50, 80, 255), (255, 220, 40), (170, 80, 255), (255, 120, 80),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8823)
    parser.add_argument("--min-gt", type=int, default=1)
    parser.add_argument("--max-gt", type=int, default=500)
    parser.add_argument(
        "--sizes", default="576x768,704x928,896x1184",
        help="Comma-separated post-preprocessing HxW sizes.")
    parser.add_argument("--no-aug", action="store_true")
    return parser.parse_args()


def valid_targets(targets):
    targets = np.asarray(targets)
    if targets.ndim != 2 or targets.shape[1] < 10:
        return np.empty((0, 10), dtype=np.float32)
    points = targets[:, 1:9].reshape(-1, 4, 2)
    extent = points.max(axis=1) - points.min(axis=1)
    keep = (extent[:, 0] > 1.0) & (extent[:, 1] > 1.0)
    return targets[keep]


def band_view(chw, group):
    hwc = np.asarray(chw).transpose(1, 2, 0)
    if group == 0:
        rgb = hwc[:, :, 0:3]
    elif group == 1:
        rgb = hwc[:, :, 3:6]
    else:
        mean = (hwc[:, :, 6] + hwc[:, :, 7]) * 0.5
        rgb = np.stack((hwc[:, :, 6], hwc[:, :, 7], mean), axis=2)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def draw_targets(image, targets, title):
    canvas = image.copy()
    for row in valid_targets(targets):
        class_id = int(round(float(row[0])))
        track_id = int(round(float(row[9])))
        points = np.rint(row[1:9].reshape(4, 2)).astype(np.int32)
        color = COLORS[class_id % len(COLORS)]
        cv2.polylines(canvas, [points], True, color, 2, cv2.LINE_AA)
        anchor = points[np.argmin(points[:, 1])].copy()
        anchor[0] = np.clip(anchor[0], 0, max(canvas.shape[1] - 1, 0))
        anchor[1] = np.clip(anchor[1] - 4, 16, max(canvas.shape[0] - 1, 16))
        class_name = (CLASS_NAMES[class_id]
                      if 0 <= class_id < len(CLASS_NAMES)
                      else "class{}".format(class_id))
        cv2.putText(canvas, "{} id{}".format(class_name, track_id),
                    tuple(anchor), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1,
                    cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (15, 15, 15), -1)
    cv2.putText(canvas, title, (9, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def make_pair(ref_image, ref_targets, cur_image, cur_targets, group, size):
    ref = draw_targets(
        band_view(ref_image, group), ref_targets,
        "REF | P{} | {}x{} | GT={}".format(
            group + 1, size[0], size[1], len(valid_targets(ref_targets))))
    cur = draw_targets(
        band_view(cur_image, group), cur_targets,
        "CUR | P{} | pair range [-2,2] | GT={}".format(
            group + 1, len(valid_targets(cur_targets))))
    return np.concatenate((ref, cur), axis=1)


def make_spectral_sheet(ref_image, ref_targets, cur_image, cur_targets, size):
    rows = []
    for name, image, targets in (
            ("REF", ref_image, ref_targets), ("CUR", cur_image, cur_targets)):
        panels = []
        for group in range(3):
            panel = draw_targets(
                band_view(image, group), targets,
                "{} P{} | GT={}".format(
                    name, group + 1, len(valid_targets(targets))))
            scale = 400.0 / panel.shape[1]
            panel = cv2.resize(
                panel, (400, int(round(panel.shape[0] * scale))),
                interpolation=cv2.INTER_AREA)
            panels.append(panel)
        rows.append(np.concatenate(panels, axis=1))
    return np.concatenate(rows, axis=0)


def main():
    args = parse_args()
    sizes = [tuple(map(int, item.split("x")))
             for item in args.sizes.split(",")]
    os.makedirs(args.output, exist_ok=True)

    exp = get_exp(args.exp_file, None)
    exp.data_num_workers = 0
    loader = exp.get_data_loader(1, False, no_aug=args.no_aug)
    dataset = loader.dataset
    index_rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    index_rng.shuffle(indices)

    metadata = []
    cursor = 0
    for sample_index in range(args.samples):
        size = sizes[sample_index % len(sizes)]
        accepted = None
        for attempt in range(100):
            index = indices[cursor % len(indices)]
            cursor += 1
            sample_seed = args.seed + sample_index * 1000 + attempt
            random.seed(sample_seed)
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            sample = dataset[(size, index, not args.no_aug)]
            ref_image, ref_targets, cur_image, cur_targets = sample[:4]
            ref_valid = valid_targets(ref_targets)
            cur_valid = valid_targets(cur_targets)
            if (args.min_gt <= len(ref_valid) <= args.max_gt and
                    args.min_gt <= len(cur_valid) <= args.max_gt):
                accepted = (index, sample_seed, ref_image, ref_targets,
                            cur_image, cur_targets, ref_valid, cur_valid)
                break
        if accepted is None:
            raise RuntimeError("could not find a non-empty pair sample")

        (index, sample_seed, ref_image, ref_targets, cur_image, cur_targets,
         ref_valid, cur_valid) = accepted
        pair = make_pair(
            ref_image, ref_targets, cur_image, cur_targets, 0, size)
        spectral = make_spectral_sheet(
            ref_image, ref_targets, cur_image, cur_targets, size)
        stem = "train_gt_{:02d}_{}x{}".format(
            sample_index + 1, size[0], size[1])
        cv2.imwrite(os.path.join(args.output, stem + "_pair.jpg"), pair,
                    [cv2.IMWRITE_JPEG_QUALITY, 94])
        cv2.imwrite(os.path.join(args.output, stem + "_spectral.jpg"), spectral,
                    [cv2.IMWRITE_JPEG_QUALITY, 94])

        ref_ids = [int(round(value)) for value in ref_valid[:, 9]]
        cur_ids = [int(round(value)) for value in cur_valid[:, 9]]
        metadata.append({
            "sample": sample_index + 1,
            "dataset_index": int(index),
            "seed": sample_seed,
            "input_hw": list(size),
            "image_shape": list(np.asarray(ref_image).shape),
            "image_dtype": str(np.asarray(ref_image).dtype),
            "image_range": [float(np.min(ref_image)),
                            float(np.max(ref_image))],
            "ref_gt_count": len(ref_valid),
            "cur_gt_count": len(cur_valid),
            "paired_track_ids_equal": ref_ids == cur_ids,
            "track_ids": ref_ids,
        })

    with open(os.path.join(args.output, "metadata.json"), "w",
              encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
