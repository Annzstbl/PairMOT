#!/usr/bin/env python3
"""Convert HSMOT COCO-video annotations to per-sequence MOT text files."""

import argparse
import collections
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    parser.add_argument("output_dir")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.json_file, "r", encoding="utf-8") as stream:
        data = json.load(stream)

    images = {image["id"]: image for image in data["images"]}
    rows = collections.defaultdict(list)
    sequences = set()
    for image in images.values():
        sequences.add(os.path.dirname(image["file_name"]))

    for annotation in data["annotations"]:
        image = images[annotation["image_id"]]
        sequence = os.path.dirname(image["file_name"])
        polygon = annotation["bbox"]
        if len(polygon) != 8:
            raise ValueError("annotation {} does not contain an 8-value qbox".
                             format(annotation.get("id")))
        # HSMOTDataset expects:
        # frame, track, x1,y1,...,x4,y4, score, zero-based class, truncation
        row = [image["frame_id"], annotation["track_id"]]
        row.extend(polygon)
        row.extend([1, int(annotation["category_id"]) - 1, 0])
        rows[sequence].append(row)

    os.makedirs(args.output_dir, exist_ok=True)
    for sequence in sorted(sequences):
        path = os.path.join(args.output_dir, sequence + ".txt")
        with open(path, "w", encoding="utf-8") as stream:
            for row in sorted(rows[sequence], key=lambda value: (value[0], value[1])):
                stream.write(",".join(str(value) for value in row) + "\n")

    print("converted {} sequences and {} annotations to {}".format(
        len(sequences), len(data["annotations"]), args.output_dir))


if __name__ == "__main__":
    main()
