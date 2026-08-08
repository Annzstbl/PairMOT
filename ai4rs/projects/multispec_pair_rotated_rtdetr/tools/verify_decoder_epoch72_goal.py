#!/usr/bin/env python3
"""Verify the strict PairMOT decoder goal at one complete epoch checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work_dir', type=Path)
    parser.add_argument('--epoch', type=int, default=72)
    parser.add_argument('--cls-threshold', type=float, default=54.437)
    parser.add_argument('--det-threshold', type=float, default=62.393)
    parser.add_argument('--sum-threshold', type=float, default=117.830)
    parser.add_argument('--stretch-sum', type=float, default=118.355)
    parser.add_argument('--expected-records', type=int, default=5416)
    parser.add_argument('--expected-sequences', type=int, default=50)
    parser.add_argument('--expected-csv', type=int, default=28)
    parser.add_argument('--expected-nonempty-files', type=int, default=108)
    parser.add_argument('--expected-predictions', type=int, default=50)
    parser.add_argument('--json-out', type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as file:
        return json.load(file)


def finite_float(mapping: dict[str, Any], key: str) -> float:
    value = float(mapping[key])
    if not math.isfinite(value):
        raise ValueError(f'{key} is not finite: {value}')
    return value


def find_epoch_eval(work_dir: Path, epoch: int) -> tuple[Path, dict[str, Any]]:
    matches = []
    for payload_path in sorted(
            (work_dir / 'val_track_eval').glob(
                'val_track_*/async_track_eval_payload.json')):
        payload = load_json(payload_path)
        if int(payload['step']) + 1 == epoch:
            matches.append((payload_path.parent, payload))
    if len(matches) != 1:
        raise RuntimeError(
            f'expected exactly one TrackEval for epoch {epoch}, got '
            f'{len(matches)}')
    return matches[0]


def load_one_row(path: Path) -> dict[str, float]:
    with path.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1:
        raise RuntimeError(f'expected one aggregate row in {path}, got {len(rows)}')
    return {key: float(value) for key, value in rows[0].items()}


def unique_file(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f'expected exactly one {pattern} below {root}, got {len(matches)}')
    return matches[0]


def verify(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = args.work_dir.resolve()
    checkpoint = work_dir / f'epoch_{args.epoch}.pth'
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(f'missing nonempty checkpoint: {checkpoint}')

    eval_dir, payload = find_epoch_eval(work_dir, args.epoch)
    track = load_json(eval_dir / 'metrics.json')
    if finite_float(track, 'track/async_done') != 1.0:
        raise RuntimeError('TrackEval has not completed successfully')

    payload_records = int(payload['num_records'])
    payload_sequences = int(payload['num_sequences'])
    track_records = int(finite_float(track, 'track/num_records'))
    track_sequences = int(finite_float(track, 'track/num_sequences'))
    if payload_records != args.expected_records or track_records != args.expected_records:
        raise RuntimeError(
            f'expected {args.expected_records} records, got payload/track '
            f'{payload_records}/{track_records}')
    if (payload_sequences != args.expected_sequences
            or track_sequences != args.expected_sequences):
        raise RuntimeError(
            f'expected {args.expected_sequences} sequences, got payload/track '
            f'{payload_sequences}/{track_sequences}')

    det_dir = Path(payload['track_kwargs']['val_det_dir'])
    det = load_json(det_dir / 'metrics.json')
    det_records = int(finite_float(det, 'val_det/num_records'))
    det_sequences = int(finite_float(det, 'val_det/num_sequences'))
    if det_records != args.expected_records or det_sequences != args.expected_sequences:
        raise RuntimeError(
            f'detection count mismatch: {det_records}/{det_sequences}')

    csv_count = sum(1 for path in eval_dir.rglob('*.csv') if path.is_file())
    nonempty_count = sum(
        1 for path in eval_dir.rglob('*')
        if path.is_file() and path.stat().st_size > 0)
    prediction_count = sum(
        1 for path in eval_dir.rglob('*.txt')
        if 'preds' in path.parts and path.stat().st_size > 0)
    expected_counts = (
        args.expected_csv,
        args.expected_nonempty_files,
        args.expected_predictions,
    )
    actual_counts = (csv_count, nonempty_count, prediction_count)
    if actual_counts != expected_counts:
        raise RuntimeError(
            'TrackEval artifact mismatch: '
            f'csv/nonempty/predictions={actual_counts}, expected={expected_counts}')

    cls_summary = load_one_row(unique_file(
        eval_dir, 'trackers/*/eval/cls_comb_cls_av_summary.csv'))
    det_summary = load_one_row(unique_file(
        eval_dir, 'trackers/*/eval/cls_comb_det_av_summary.csv'))
    cls_hota = finite_float(track, 'track/cls_hota')
    det_hota = finite_float(track, 'track/det_hota')
    cls_decimal = Decimal(str(cls_hota))
    det_decimal = Decimal(str(det_hota))
    hota_sum_decimal = cls_decimal + det_decimal
    cls_threshold = Decimal(str(args.cls_threshold))
    det_threshold = Decimal(str(args.det_threshold))
    sum_threshold = Decimal(str(args.sum_threshold))
    stretch_sum = Decimal(str(args.stretch_sum))
    hota_sum = float(hota_sum_decimal)
    fallback_pass = (
        cls_decimal > cls_threshold
        and det_decimal > det_threshold
        and hota_sum_decimal > sum_threshold)
    stretch_pass = hota_sum_decimal >= stretch_sum

    result = {
        'work_dir': str(work_dir),
        'epoch': args.epoch,
        'checkpoint': str(checkpoint),
        'checkpoint_bytes': checkpoint.stat().st_size,
        'eval_dir': str(eval_dir),
        'cls_hota': cls_hota,
        'det_hota': det_hota,
        'hota_sum': hota_sum,
        'cls_deta': float(cls_summary['DetA']),
        'cls_assa': float(cls_summary['AssA']),
        'det_deta': float(det_summary['DetA']),
        'det_assa': float(det_summary['AssA']),
        'pair_map': finite_float(det, 'pair_mAP50_95'),
        'pair_ap50': finite_float(det, 'pair_AP50'),
        'both_independent_map': finite_float(det, 'both_independent_mAP50_95'),
        'both_independent_ap50': finite_float(det, 'both_independent_AP50'),
        'csv_files': csv_count,
        'nonempty_files': nonempty_count,
        'nonempty_predictions': prediction_count,
        'fallback_pass': fallback_pass,
        'stretch_pass': stretch_pass,
        'margins': {
            'cls': float(cls_decimal - cls_threshold),
            'det': float(det_decimal - det_threshold),
            'sum': float(hota_sum_decimal - sum_threshold),
            'stretch_sum': float(hota_sum_decimal - stretch_sum),
        },
    }
    return result


def main() -> None:
    args = parse_args()
    result = verify(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open('w', encoding='utf-8') as file:
            json.dump(result, file, indent=2, sort_keys=True)
            file.write('\n')
    if not result['fallback_pass']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
