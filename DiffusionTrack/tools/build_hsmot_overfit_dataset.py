#!/usr/bin/env python3
"""Build an HSMOT detection overfit split from one frame repeated N times."""

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", required=True,
        help="Source HxWx8 NPY frame or the _p1.jpg part of a 3-JPG frame")
    parser.add_argument(
        "--image-format", choices=("npy", "3jpg"), default="npy")
    parser.add_argument(
        "--image-subdir", default=None,
        help="Output image subdirectory (defaults to npy or npy2jpg)")
    parser.add_argument("--annotation", required=True,
                        help="Source sequence-level 13-column MOT annotation")
    parser.add_argument("--source-frame", type=int, default=1)
    parser.add_argument("--sequence", default="overfit-one-image")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    image = Path(args.image).resolve(strict=True)
    annotation = Path(args.annotation).resolve(strict=True)
    output = Path(args.output).resolve()
    image_subdir = args.image_subdir or (
        "npy" if args.image_format == "npy" else "npy2jpg")
    image_dir = output / image_subdir / args.sequence
    annotation_dir = output / "mot"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    source_rows = []
    for line in annotation.read_text(encoding="utf-8").splitlines():
        values = line.split(",")
        if values and int(float(values[0])) == args.source_frame:
            source_rows.append(values)
    if not source_rows:
        raise ValueError(
            f"frame {args.source_frame} has no annotations in {annotation}")

    repeated_rows = []
    if args.image_format == "3jpg":
        suffix = "_p1.jpg"
        if not image.name.endswith(suffix):
            raise ValueError("3jpg --image must end with _p1.jpg")
        stem = image.name[:-len(suffix)]
        image_parts = [
            image.with_name(f"{stem}_p{part}.jpg").resolve(strict=True)
            for part in (1, 2, 3)]
    else:
        image_parts = [image]

    for frame_id in range(1, args.repeat + 1):
        for part_index, source in enumerate(image_parts, start=1):
            filename = (f"{frame_id:06d}.npy"
                        if args.image_format == "npy" else
                        f"{frame_id:06d}_p{part_index}.jpg")
            destination = image_dir / filename
            if destination.is_symlink() or destination.exists():
                destination.unlink()
            os.symlink(source, destination)
        for values in source_rows:
            row = values.copy()
            row[0] = str(frame_id)
            repeated_rows.append(",".join(row))

    annotation_path = annotation_dir / f"{args.sequence}.txt"
    annotation_path.write_text(
        "\n".join(repeated_rows) + "\n", encoding="utf-8")
    print(f"dataset={output}")
    print(f"image={image}")
    print(f"image_format={args.image_format}")
    print(f"frames={args.repeat}, objects_per_frame={len(source_rows)}")
    print(f"annotation={annotation_path}")


if __name__ == "__main__":
    main()
