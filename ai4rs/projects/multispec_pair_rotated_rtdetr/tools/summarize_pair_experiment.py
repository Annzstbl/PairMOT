#!/usr/bin/env python3
"""Select and summarize a PairMOT checkpoint from asynchronous validation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PAPER_BASE = {
    'cls_hota': 53.314,
    'det_hota': 61.982,
    'cls_deta': 43.386,
    'cls_assa': 68.287,
    'det_deta': 53.890,
    'det_assa': 73.643,
}


@dataclass(frozen=True)
class EvalPoint:
    epoch: int
    eval_dir: Path
    track: dict[str, Any]
    det: dict[str, Any]
    cls_summary: dict[str, float]
    det_summary: dict[str, float]

    @property
    def hota_sum(self) -> float:
        return float(self.track['track/cls_hota']
                     + self.track['track/det_hota'])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work_dir', type=Path)
    parser.add_argument('--expected-evals', type=int, default=18)
    parser.add_argument(
        '--allow-partial',
        action='store_true',
        help='Summarize available points without claiming a formal best.')
    parser.add_argument('--json-out', type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as file:
        return json.load(file)


def load_summary(path: Path) -> dict[str, float]:
    with path.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1:
        raise ValueError(f'expected one aggregate row in {path}, got {len(rows)}')
    return {key: float(value) for key, value in rows[0].items()}


def locate_eval_csv(eval_dir: Path, filename: str) -> Path:
    matches = list(eval_dir.glob(f'trackers/*/eval/{filename}'))
    if len(matches) != 1:
        raise FileNotFoundError(
            f'expected one {filename} below {eval_dir}, got {len(matches)}')
    return matches[0]


def load_point(work_dir: Path, eval_dir: Path) -> EvalPoint:
    payload = load_json(eval_dir / 'async_track_eval_payload.json')
    step = int(payload['step'])
    epoch = step + 1
    det_dir = work_dir / 'val_det' / f'epoch_{step:02d}'
    track = load_json(eval_dir / 'metrics.json')
    det = load_json(det_dir / 'metrics.json')
    cls_summary = load_summary(
        locate_eval_csv(eval_dir, 'cls_comb_cls_av_summary.csv'))
    det_summary = load_summary(
        locate_eval_csv(eval_dir, 'cls_comb_det_av_summary.csv'))
    return EvalPoint(
        epoch=epoch,
        eval_dir=eval_dir,
        track=track,
        det=det,
        cls_summary=cls_summary,
        det_summary=det_summary)


def collect_points(work_dir: Path) -> list[EvalPoint]:
    eval_root = work_dir / 'val_track_eval'
    points = []
    errors = []
    for eval_dir in sorted(eval_root.glob('val_track_*')):
        try:
            points.append(load_point(work_dir, eval_dir))
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
            errors.append(f'{eval_dir.name}: {error}')
    if errors:
        raise RuntimeError('incomplete evaluation artifacts:\n  ' +
                           '\n  '.join(errors))
    epochs = [point.epoch for point in points]
    if len(epochs) != len(set(epochs)):
        raise RuntimeError(f'duplicate evaluated epochs: {epochs}')
    return sorted(points, key=lambda point: point.epoch)


def select_best(points: list[EvalPoint]) -> EvalPoint:
    if not points:
        raise RuntimeError('no complete TrackEval points found')
    best_sum = max(point.hota_sum for point in points)
    winners = [
        point for point in points if abs(point.hota_sum - best_sum) < 1e-12
    ]
    if len(winners) != 1:
        epochs = [point.epoch for point in winners]
        raise RuntimeError(f'HOTA-sum best is not unique: epochs={epochs}')
    return winners[0]


def result_dict(
        work_dir: Path,
        points: list[EvalPoint],
        best: EvalPoint,
        complete: bool,
) -> dict[str, Any]:
    track = best.track
    result = {
        'work_dir': str(work_dir),
        'status': 'complete' if complete else 'partial',
        'num_evals': len(points),
        'evaluated_epochs': [point.epoch for point in points],
        'best_epoch': best.epoch,
        'cls_hota': track['track/cls_hota'],
        'det_hota': track['track/det_hota'],
        'hota_sum': best.hota_sum,
        'cls_deta': best.cls_summary['DetA'],
        'cls_assa': best.cls_summary['AssA'],
        'cls_mota': track['track/cls_mota'],
        'cls_idf1': track['track/cls_idf1'],
        'det_deta': best.det_summary['DetA'],
        'det_assa': best.det_summary['AssA'],
        'det_mota': track['track/det_mota'],
        'det_idf1': track['track/det_idf1'],
        'pair_map': best.det['pair_mAP50_95'],
        'pair_ap50': best.det['pair_AP50'],
    }
    result['delta_vs_paper_base'] = {
        key: result[key] - value for key, value in PAPER_BASE.items()
    }
    return result


def print_summary(result: dict[str, Any]) -> None:
    delta = result['delta_vs_paper_base']
    print(f"status={result['status']} evals={result['num_evals']} "
          f"best_epoch={result['best_epoch']}")
    print(f"cls HOTA={result['cls_hota']:.3f} "
          f"DetA={result['cls_deta']:.3f} "
          f"AssA={result['cls_assa']:.3f} "
          f"MOTA={result['cls_mota']:.3f} "
          f"IDF1={result['cls_idf1']:.3f}")
    print(f"det HOTA={result['det_hota']:.3f} "
          f"DetA={result['det_deta']:.3f} "
          f"AssA={result['det_assa']:.3f} "
          f"MOTA={result['det_mota']:.3f} "
          f"IDF1={result['det_idf1']:.3f}")
    print(f"pair mAP={result['pair_map']:.4f} "
          f"AP50={result['pair_ap50']:.4f}")
    print('delta vs Paper Base: '
          f"cls HOTA={delta['cls_hota']:+.3f}, "
          f"det HOTA={delta['det_hota']:+.3f}, "
          f"det DetA={delta['det_deta']:+.3f}, "
          f"det AssA={delta['det_assa']:+.3f}")


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    points = collect_points(work_dir)
    complete = len(points) == args.expected_evals
    if not complete and not args.allow_partial:
        raise RuntimeError(
            f'expected {args.expected_evals} complete evaluations, '
            f'found {len(points)}; use --allow-partial for diagnostics')
    best = select_best(points)
    result = result_dict(work_dir, points, best, complete)
    print_summary(result)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open('w', encoding='utf-8') as file:
            json.dump(result, file, indent=2, sort_keys=True)
            file.write('\n')


if __name__ == '__main__':
    main()
