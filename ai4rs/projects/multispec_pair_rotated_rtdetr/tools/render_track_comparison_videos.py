#!/usr/bin/env python3
"""Render one full-resolution video per sequence and result."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = (
    'car', 'bike', 'pedestrian', 'van', 'truck', 'bus', 'tricycle',
    'awning-bike')
COLORS = {
    'GT': (0, 190, 255),
    'BASE': (40, 210, 40),
    '0719_02': (255, 130, 20),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--gt-root', type=Path, required=True)
    parser.add_argument('--base-root', type=Path, required=True)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--metrics-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=8.0)
    parser.add_argument('--codec', default='mp4v')
    parser.add_argument('--sequences', nargs='*')
    parser.add_argument('--base-name', default='BASE')
    parser.add_argument('--experiment-name', default='0719_02')
    return parser.parse_args()


def read_mot(path: Path) -> dict[int, list[dict]]:
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
                'cls': int(values[11]),
            })
    return frames


def read_metrics(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return {row['sequence']: row for row in csv.DictReader(handle)}


def winner(delta: float, experiment_name: str, base_name: str) -> str:
    if abs(delta) < 0.05:
        return 'TIE'
    return experiment_name if delta > 0 else base_name


def draw_item(image: np.ndarray, item: dict, color: tuple[int, int, int],
              show_class: bool) -> None:
    points = np.round(item['poly']).astype(np.int32)
    cv2.polylines(image, [points], True, color, 3, cv2.LINE_AA)
    anchor = tuple(points[np.argmin(points[:, 1])].astype(int))
    label = str(item['id'])
    if show_class:
        cls_index = item['cls']
        cls_name = CLASS_NAMES[cls_index] if 0 <= cls_index < 8 else str(cls_index)
        label = f'{label}:{cls_name}'
    (width, height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    x = max(0, min(anchor[0], image.shape[1] - width - 5))
    y = max(height + 5, min(anchor[1] - 3, image.shape[0] - baseline - 2))
    cv2.rectangle(image, (x - 2, y - height - 3),
                  (x + width + 2, y + baseline + 2), (20, 20, 20), -1)
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                color, 1, cv2.LINE_AA)


def add_header(image: np.ndarray, sequence: str, result_name: str,
               frame: int, metrics: dict[str, str], experiment_name: str,
               base_name: str) -> np.ndarray:
    header_height = 92
    canvas = np.full((image.shape[0] + header_height, image.shape[1], 3),
                     242, dtype=np.uint8)
    canvas[header_height:] = image
    det_delta = float(metrics['det_HOTA_delta'])
    cls_delta = float(metrics['cls_HOTA_delta'])
    det_a_delta = float(metrics['DetA_delta'])
    ass_a_delta = float(metrics['AssA_delta'])
    title = f'{sequence} | {result_name} | frame {frame}'
    subtitle = (
        f'det winner: {winner(det_delta, experiment_name, base_name)} '
        f'({det_delta:+.2f})  |  cls winner: '
        f'{winner(cls_delta, experiment_name, base_name)} ({cls_delta:+.2f})  |  '
        f'DetA {det_a_delta:+.2f}  AssA {ass_a_delta:+.2f}')
    cv2.putText(canvas, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.76,
                COLORS[result_name], 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def image_files(image_root: Path, sequence: str) -> list[Path]:
    return sorted((image_root / sequence).glob('*_p2.jpg'))


def frame_number(path: Path) -> int:
    return int(path.stem.split('_')[0])


def open_writer(path: Path, codec: str, fps: float,
                frame_size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*codec), fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f'Cannot open video writer for {path} with {codec}')
    return writer


def main() -> None:
    args = parse_args()
    metrics = read_metrics(args.metrics_csv)
    COLORS[args.base_name] = COLORS['BASE']
    COLORS[args.experiment_name] = COLORS['0719_02']
    sequences = args.sequences or sorted(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sequence_index, sequence in enumerate(sequences, 1):
        files = image_files(args.image_root, sequence)
        if not files:
            raise FileNotFoundError(f'No p2 images found for {sequence}')
        tracks = {
            'GT': read_mot(args.gt_root / f'{sequence}.txt'),
            args.base_name: read_mot(args.base_root / f'{sequence}.txt'),
            args.experiment_name: read_mot(
                args.experiment_root / f'{sequence}.txt'),
        }
        sample = cv2.imread(str(files[0]))
        if sample is None:
            raise RuntimeError(f'Cannot read {files[0]}')
        height, width = sample.shape[:2]
        sequence_dir = args.output_dir / sequence
        sequence_dir.mkdir(parents=True, exist_ok=True)
        writers = {
            result_name: open_writer(
                sequence_dir / f'{sequence}_{result_name}.mp4', args.codec,
                args.fps, (width, height + 92))
            for result_name in tracks
        }
        try:
            for path in files:
                frame = frame_number(path)
                source = cv2.imread(str(path))
                if source is None:
                    raise RuntimeError(f'Cannot read {path}')
                for result_name, frame_tracks in tracks.items():
                    rendered = source.copy()
                    for item in frame_tracks.get(frame, []):
                        draw_item(rendered, item, COLORS[result_name],
                                  show_class=result_name == 'GT')
                    writers[result_name].write(add_header(
                        rendered, sequence, result_name, frame,
                        metrics[sequence], args.experiment_name,
                        args.base_name))
        finally:
            for writer in writers.values():
                writer.release()
        print(f'[{sequence_index:02d}/{len(sequences):02d}] {sequence}: '
              f'{len(files)} frames', flush=True)


if __name__ == '__main__':
    main()
