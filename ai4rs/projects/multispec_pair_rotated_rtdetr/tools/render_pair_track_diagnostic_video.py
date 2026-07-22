#!/usr/bin/env python3
"""Render side-by-side pair detections and previous-frame active tracks."""

from __future__ import annotations

import argparse
import colorsys
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


HEADER_HEIGHT = 110
PANEL_GAP = 24
TRACK_COLOR = (255, 30, 220)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--result-name', required=True)
    parser.add_argument('--pair-det', type=Path, required=True)
    parser.add_argument('--tracks', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--track-th', type=float, default=0.2)
    parser.add_argument('--new-born-th', type=float, default=0.6)
    parser.add_argument('--fps', type=float, default=4.0)
    return parser.parse_args()


def polygon(prefix: str, row: dict[str, str]) -> np.ndarray:
    values = []
    for index in range(1, 5):
        values.extend((float(row[f'{prefix}_x{index}']),
                       float(row[f'{prefix}_y{index}'])))
    return np.asarray(values, dtype=np.float32).reshape(4, 2)


def read_pair_detections(path: Path) -> dict[tuple[int, int], list[dict]]:
    pairs = defaultdict(list)
    with path.open(encoding='utf-8') as handle:
        header = handle.readline().lstrip('#').strip().split(',')
        for row in csv.DictReader(handle, fieldnames=header):
            record = {
                'prev_frame': int(row['prev_frame']),
                'curr_frame': int(row['curr_frame']),
                'index': int(row['det_index']),
                'prev_poly': polygon('prev', row),
                'curr_poly': polygon('curr', row),
                'prev_score': float(row['prev_score']),
                'curr_score': float(row['curr_score']),
                'pair_score': float(row['pair_score']),
            }
            pairs[(record['prev_frame'], record['curr_frame'])].append(record)
    return pairs


def read_tracks(path: Path) -> dict[int, list[dict]]:
    frames = defaultdict(list)
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip() or line.startswith('#'):
                continue
            values = [float(value) for value in line.strip().split(',')]
            frames[int(values[0])].append({
                'id': int(values[1]),
                'poly': np.asarray(values[2:10], dtype=np.float32).reshape(4, 2),
                'score': float(values[10]),
            })
    return frames


def pair_color(index: int) -> tuple[int, int, int]:
    hue = (index * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.88, 1.0)
    return int(blue * 255), int(green * 255), int(red * 255)


def center(poly: np.ndarray, x_offset: int = 0) -> tuple[int, int]:
    point = poly.mean(axis=0)
    return int(point[0]) + x_offset, int(point[1]) + HEADER_HEIGHT


def draw_label(canvas: np.ndarray, text: str, anchor: tuple[int, int],
               color: tuple[int, int, int], occupied: list[tuple[int, int, int, int]],
               panel_min_x: int, panel_max_x: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    (width, height), baseline = cv2.getTextSize(text, font, scale, 1)
    candidates = [
        (anchor[0] + 4, anchor[1] - 5),
        (anchor[0] + 4, anchor[1] + height + 7),
        (anchor[0] - width - 5, anchor[1] - 5),
        (anchor[0] - width - 5, anchor[1] + height + 7),
    ]
    selected = None
    for x, y in candidates:
        x = max(panel_min_x + 2, min(x, panel_max_x - width - 5))
        y = max(HEADER_HEIGHT + height + 3,
                min(y, canvas.shape[0] - baseline - 3))
        rect = (x - 2, y - height - 3, x + width + 2, y + baseline + 2)
        if not any(rect[0] < other[2] and rect[2] > other[0] and
                   rect[1] < other[3] and rect[3] > other[1]
                   for other in occupied):
            selected = (x, y, rect)
            break
    if selected is None:
        x, y = candidates[0]
        x = max(panel_min_x + 2, min(x, panel_max_x - width - 5))
        y = max(HEADER_HEIGHT + height + 3,
                min(y, canvas.shape[0] - baseline - 3))
        selected = (x, y, (x - 2, y - height - 3,
                           x + width + 2, y + baseline + 2))
    x, y, rect = selected
    occupied.append(rect)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]),
                  (12, 12, 12), -1)
    cv2.putText(canvas, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


def draw_polygon(canvas: np.ndarray, poly: np.ndarray, x_offset: int,
                 color: tuple[int, int, int], thickness: int) -> np.ndarray:
    points = np.round(poly).astype(np.int32)
    points[:, 0] += x_offset
    points[:, 1] += HEADER_HEIGHT
    cv2.polylines(canvas, [points], True, color, thickness, cv2.LINE_AA)
    return points


def image_path(root: Path, sequence: str, frame: int) -> Path:
    return root / sequence / f'{frame:06d}_p2.jpg'


def main() -> None:
    args = parse_args()
    pairs = read_pair_detections(args.pair_det)
    tracks = read_tracks(args.tracks)
    transitions = sorted(pairs)
    if not transitions:
        raise RuntimeError(f'No detections in {args.pair_det}')

    first_image = cv2.imread(str(image_path(
        args.image_root, args.sequence, transitions[0][0])))
    if first_image is None:
        raise RuntimeError('Cannot read first source frame')
    height, width = first_image.shape[:2]
    output_size = (width * 2 + PANEL_GAP, height + HEADER_HEIGHT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), args.fps,
        output_size)
    if not writer.isOpened():
        raise RuntimeError(f'Cannot open {args.output}')

    try:
        for prev_frame, curr_frame in transitions:
            prev_image = cv2.imread(str(image_path(
                args.image_root, args.sequence, prev_frame)))
            curr_image = cv2.imread(str(image_path(
                args.image_root, args.sequence, curr_frame)))
            if prev_image is None or curr_image is None:
                raise RuntimeError(f'Missing images for {prev_frame}->{curr_frame}')
            canvas = np.full((output_size[1], output_size[0], 3), 242,
                             dtype=np.uint8)
            canvas[HEADER_HEIGHT:, :width] = prev_image
            canvas[HEADER_HEIGHT:, width + PANEL_GAP:] = curr_image
            cv2.rectangle(canvas, (width, HEADER_HEIGHT),
                          (width + PANEL_GAP - 1, output_size[1] - 1),
                          (35, 35, 35), -1)

            relevant = [
                detection for detection in pairs[(prev_frame, curr_frame)]
                if detection['prev_score'] >= args.track_th or
                detection['curr_score'] >= args.new_born_th
            ]
            title = (f'{args.sequence} | {args.result_name} | '
                     f'pair {prev_frame} -> {curr_frame}')
            subtitle = (
                f'LEFT: prev detections + active tracks   RIGHT: curr detections   '
                f'pair detections shown: {len(relevant)} / 300   '
                f'rule: prev>={args.track_th:.1f} or curr>={args.new_born_th:.1f}')
            cv2.putText(canvas, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.82, (25, 25, 25), 2, cv2.LINE_AA)
            cv2.putText(canvas, subtitle, (18, 72), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (35, 35, 35), 1, cv2.LINE_AA)
            cv2.line(canvas, (18, 94), (52, 94), TRACK_COLOR, 4, cv2.LINE_AA)
            cv2.putText(canvas, 'active track (TID, score)', (60, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (35, 35, 35), 1,
                        cv2.LINE_AA)

            # Draw pair links first so boxes and labels remain readable.
            for detection in relevant:
                color = pair_color(detection['index'])
                cv2.line(canvas, center(detection['prev_poly']),
                         center(detection['curr_poly'], width + PANEL_GAP),
                         color, 1, cv2.LINE_AA)

            occupied_left: list[tuple[int, int, int, int]] = []
            occupied_right: list[tuple[int, int, int, int]] = []
            for detection in relevant:
                color = pair_color(detection['index'])
                prev_points = draw_polygon(
                    canvas, detection['prev_poly'], 0, color, 2)
                curr_points = draw_polygon(
                    canvas, detection['curr_poly'], width + PANEL_GAP,
                    color, 2)
                prev_anchor = tuple(prev_points[np.argmin(prev_points[:, 1])])
                curr_anchor = tuple(curr_points[np.argmin(curr_points[:, 1])])
                draw_label(
                    canvas,
                    f'D{detection["index"]} p={detection["prev_score"]:.3f}',
                    prev_anchor, color, occupied_left, 0, width)
                draw_label(
                    canvas,
                    f'D{detection["index"]} c={detection["curr_score"]:.3f}',
                    curr_anchor, color, occupied_right,
                    width + PANEL_GAP, output_size[0])

            # Active means the track has an updated output at the previous frame.
            for track in tracks.get(prev_frame, []):
                points = draw_polygon(canvas, track['poly'], 0,
                                      TRACK_COLOR, 4)
                anchor = tuple(points[np.argmin(points[:, 1])])
                draw_label(canvas,
                           f'T{track["id"]} s={track["score"]:.3f}', anchor,
                           TRACK_COLOR, occupied_left, 0, width)
            writer.write(canvas)
    finally:
        writer.release()

    print(f'{args.output}: {len(transitions)} transitions, '
          f'{output_size[0]}x{output_size[1]}')


if __name__ == '__main__':
    main()
