#!/usr/bin/env python3
"""Build a deterministic COCO subset by repeating one source image."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    source = json.loads(Path(args.source).read_text())
    matches = [
        image for image in source["images"]
        if image["file_name"] == args.file_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one image named {args.file_name!r}, found {len(matches)}")
    source_image = matches[0]
    source_annotations = [
        annotation for annotation in source["annotations"]
        if annotation["image_id"] == source_image["id"]
    ]

    images = []
    annotations = []
    annotation_id = 1
    for image_id in range(1, args.repeats + 1):
        image = dict(source_image)
        image.update({
            "id": image_id,
            "video_id": 1,
            "frame_id": image_id,
            "prev_image_id": image_id - 1 if image_id > 1 else -1,
            "next_image_id": image_id + 1 if image_id < args.repeats else -1,
        })
        images.append(image)
        for source_annotation in source_annotations:
            annotation = dict(source_annotation)
            annotation.update({
                "id": annotation_id,
                "image_id": image_id,
            })
            annotations.append(annotation)
            annotation_id += 1

    result = {
        "images": images,
        "categories": source["categories"],
        "annotations": annotations,
        "videos": [{"id": 1, "file_name": "data43-2"}],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(
        f"wrote {len(images)} images and {len(annotations)} annotations "
        f"to {output}")


if __name__ == "__main__":
    main()
