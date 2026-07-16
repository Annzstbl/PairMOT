#!/usr/bin/env python3
"""Summarize a completed PairMOT run from asynchronous TrackEval outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


CORE_KEYS = (
    'track/cls_hota', 'track/cls_mota', 'track/cls_idf1',
    'track/det_hota', 'track/det_mota', 'track/det_idf1')


def load_json(path: Path) -> dict:
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def fmt(value: float | None) -> str:
    return '-' if value is None else f'{value:.3f}'


def class_hota(metrics: dict) -> dict[str, float]:
    result = {}
    pattern = re.compile(r'^track_class/(.+)_hota$')
    for key, value in metrics.items():
        match = pattern.match(key)
        if match and isinstance(value, (int, float)):
            result[match.group(1)] = float(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--expected-evals', type=int, default=18)
    parser.add_argument('--val-interval', type=int, default=4)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output-md', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()

    eval_root = args.work_dir / 'val_track_eval'
    rows = []
    for metrics_path in sorted(eval_root.glob('val_track_*/metrics.json')):
        match = re.search(r'val_track_(\d+)', metrics_path.parent.name)
        if not match:
            continue
        index = int(match.group(1))
        metrics = load_json(metrics_path)
        if float(metrics.get('track/async_done', 0.0)) != 1.0:
            continue
        if any(key not in metrics or not math.isfinite(float(metrics[key]))
               for key in CORE_KEYS):
            raise RuntimeError(f'Incomplete metrics: {metrics_path}')
        row = {
            'eval_index': index,
            'epoch': index * args.val_interval,
            **{key: float(metrics[key]) for key in CORE_KEYS},
            'class_hota': class_hota(metrics),
            'metrics_path': str(metrics_path.relative_to(args.work_dir)),
        }
        row['hota_sum'] = row['track/cls_hota'] + row['track/det_hota']
        rows.append(row)

    if len(rows) != args.expected_evals:
        raise RuntimeError(
            f'Expected {args.expected_evals} completed TrackEval outputs, '
            f'found {len(rows)}')
    rows.sort(key=lambda row: row['epoch'])
    best_score = max(row['hota_sum'] for row in rows)
    winners = [row for row in rows if row['hota_sum'] == best_score]
    if len(winners) != 1:
        raise RuntimeError(
            f'Best cls_HOTA + det_HOTA is not unique: '
            f'{[row["epoch"] for row in winners]}')
    best = winners[0]
    final = rows[-1]

    baseline = load_json(args.baseline) if args.baseline and args.baseline.exists() else None
    baseline_metrics = baseline.get('metrics', {}) if baseline else {}
    baseline_classes = baseline.get('class_hota', {}) if baseline else {}

    result = {
        'experiment_id': args.experiment_id,
        'experiment_name': args.experiment_name,
        'selection_rule': 'unique maximum of cls_HOTA + det_HOTA',
        'completed_evals': len(rows),
        'best': best,
        'final': final,
        'evaluations': rows,
        'baseline': baseline,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    lines = [
        f'# {args.experiment_id} AutoDL Result', '',
        '## Experiment', '',
        f'- Name: `{args.experiment_name}`',
        '- Protocol: full HSMOT, ordered `t-1 -> t` pairs, 1200x900, '
        'R18 COCO-adapted initialization, BF16, two GPUs.',
        f'- Completed asynchronous TrackEval points: `{len(rows)}/{args.expected_evals}`.',
        '- Checkpoint selection: unique maximum of `cls_HOTA + det_HOTA`.', '',
        '## Tracking Results', '',
        '| epoch | cls HOTA | cls MOTA | cls IDF1 | det HOTA | det MOTA | det IDF1 | HOTA sum |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(
            f'| {row["epoch"]} | {fmt(row["track/cls_hota"])} | '
            f'{fmt(row["track/cls_mota"])} | {fmt(row["track/cls_idf1"])} | '
            f'{fmt(row["track/det_hota"])} | {fmt(row["track/det_mota"])} | '
            f'{fmt(row["track/det_idf1"])} | {fmt(row["hota_sum"])} |')

    lines.extend(['', '## Selected Checkpoint', '',
                  f'- Best epoch: `{best["epoch"]}`.',
                  f'- `cls_HOTA={best["track/cls_hota"]:.3f}`.',
                  f'- `det_HOTA={best["track/det_hota"]:.3f}`.',
                  f'- `cls_HOTA + det_HOTA={best["hota_sum"]:.3f}`.',
                  f'- Final epoch sum: `{final["hota_sum"]:.3f}`; best-to-final '
                  f'delta: `{best["hota_sum"] - final["hota_sum"]:+.3f}`.', ''])

    lines.extend(['## Per-Class HOTA At Selected Epoch', '',
                  '| class | HOTA | baseline | delta |',
                  '| --- | ---: | ---: | ---: |'])
    for name, value in sorted(best['class_hota'].items()):
        base = baseline_classes.get(name)
        delta = value - float(base) if base is not None else None
        lines.append(f'| {name} | {fmt(value)} | {fmt(base)} | '
                     f'{fmt(delta) if delta is None else f"{delta:+.3f}"} |')

    lines.extend(['', '## Conclusion', ''])
    if baseline_metrics:
        base_cls = float(baseline_metrics['track/cls_hota'])
        base_det = float(baseline_metrics['track/det_hota'])
        delta_cls = best['track/cls_hota'] - base_cls
        delta_det = best['track/det_hota'] - base_det
        delta_sum = best['hota_sum'] - (base_cls + base_det)
        lines.append(
            f'Against `{baseline.get("experiment_id", "baseline")}`, '
            f'cls HOTA changes by `{delta_cls:+.3f}`, det HOTA by '
            f'`{delta_det:+.3f}`, and their sum by `{delta_sum:+.3f}`.')
        if delta_cls > 0 and delta_det > 0:
            lines.append(
                'Both primary HOTA axes improve, so Set-Transport is a '
                'positive candidate under this protocol.')
        else:
            lines.append(
                'Both primary HOTA axes do not improve simultaneously; '
                'Set-Transport should not replace the mainline Liquid model.')
    else:
        lines.append(
            'No same-protocol `0716_04` baseline JSON was available at report '
            'generation time. This run has a valid selected checkpoint, but '
            'the report does not claim that Set-Transport improves the '
            'mainline Liquid model.')
    args.output_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
