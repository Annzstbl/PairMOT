#!/usr/bin/env python3
"""Cross-environment forward/backward check for rotated FPN ROIAlign."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("lx", "mmcv"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--clockwise",
        type=int,
        choices=(0, 1),
        default=0,
        help="Only used by the MMCV backend.",
    )
    return parser.parse_args()


def make_input(path):
    rng = np.random.RandomState(20260724)
    arrays = {
        "f0": rng.standard_normal((1, 2, 128, 128)).astype("float32"),
        "f1": rng.standard_normal((1, 2, 64, 64)).astype("float32"),
        "f2": rng.standard_normal((1, 2, 32, 32)).astype("float32"),
        "boxes": np.asarray(
            [
                [120.0, 100.0, 28.0, 12.0, 32.0],
                [330.0, 190.0, 90.0, 22.0, -61.0],
                [230.0, 350.0, 150.0, 70.0, 77.0],
                [540.0, 510.0, 520.0, 310.0, -18.0],
                [515.0, 530.0, 920.0, 710.0, 123.0],
            ],
            dtype="float32",
        ),
        "output_weight": rng.standard_normal((5, 2, 7, 7)).astype("float32"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        make_input(input_path)
    data = np.load(input_path)

    device = torch.device("cuda")
    features = [
        torch.from_numpy(data[f"f{i}"]).to(device).requires_grad_(True)
        for i in range(3)
    ]
    boxes = torch.from_numpy(data["boxes"]).to(device)

    if args.backend == "lx":
        from detectron2.modeling.poolers import ROIPooler
        from detectron2.structures import RotatedBoxes

        pooler = ROIPooler(
            output_size=7,
            scales=(1 / 8, 1 / 16, 1 / 32),
            sampling_ratio=2,
            pooler_type="ROIAlignRotated",
        ).to(device)
        output = pooler(features, [RotatedBoxes(boxes)])
    else:
        from diffusion.models.diffusion_models import RotatedROIPooler

        # The shared fixture stores Detectron2/LX boxes in degrees. MMCV's
        # low-level operator consumes radians, matching our model's native
        # LE135 representation.
        boxes = boxes.clone()
        boxes[:, 4] = torch.deg2rad(boxes[:, 4])
        pooler = RotatedROIPooler(
            output_size=7,
            scales=(1 / 8, 1 / 16, 1 / 32),
            sampling_ratio=2,
        ).to(device)
        pooler.clockwise = bool(args.clockwise)
        output = pooler(features, [boxes])

    output_weight = torch.from_numpy(data["output_weight"]).to(device)
    (output * output_weight).sum().backward()
    result = {"out": output.detach().cpu().numpy()}
    result.update({
        f"g{i}": (
            feature.grad.detach().cpu().numpy()
            if feature.grad is not None
            else np.zeros_like(data[f"f{i}"])
        )
        for i, feature in enumerate(features)
    })
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **result)


if __name__ == "__main__":
    main()
