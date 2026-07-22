#!/usr/bin/env python3
"""Compare two PairMOT TrackEval runs and visualize every sequence."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


CLASS_NAMES = (
    'car', 'bike', 'pedestrian', 'van', 'truck', 'bus', 'tricycle',
    'awning-bike')


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def read_mot(path: Path) -> dict[int, list[dict]]:
    frames = defaultdict(list)
    if not path.exists():
        return frames
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


def read_pair_detections(path: Path) -> list[dict]:
    records = []
    with path.open(encoding='utf-8') as handle:
        header = handle.readline().lstrip('#').strip().split(',')
        for row in csv.DictReader(handle, fieldnames=header):
            def poly(prefix: str) -> np.ndarray:
                values = []
                for index in range(1, 5):
                    values.extend((float(row[f'{prefix}_x{index}']),
                                   float(row[f'{prefix}_y{index}'])))
                return np.asarray(values, dtype=np.float32).reshape(4, 2)

            records.append({
                'curr_frame': int(row['curr_frame']),
                'prev_frame': int(row['prev_frame']),
                'index': int(row['det_index']),
                'prev_poly': poly('prev'),
                'curr_poly': poly('curr'),
                'prev_cls': int(row['prev_cls']),
                'curr_cls': int(row['curr_cls']),
                'prev_score': float(row['prev_score']),
                'curr_score': float(row['curr_score']),
            })
    return records


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = cv2.convexHull(first).reshape(-1, 2)
    second = cv2.convexHull(second).reshape(-1, 2)
    area_first = abs(cv2.contourArea(first))
    area_second = abs(cv2.contourArea(second))
    if area_first <= 0 or area_second <= 0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = area_first + area_second - intersection
    return float(intersection / union) if union > 0 else 0.0


def match_predictions(predictions: list[dict], ground_truth: list[dict],
                      iou_threshold: float) -> tuple[dict[int, int], int, int, int, float]:
    if not predictions or not ground_truth:
        return {}, 0, len(predictions), len(ground_truth), 0.0
    costs = np.ones((len(predictions), len(ground_truth)), dtype=np.float32)
    for pred_index, prediction in enumerate(predictions):
        for gt_index, target in enumerate(ground_truth):
            costs[pred_index, gt_index] = 1.0 - polygon_iou(
                prediction['poly'], target['poly'])
    pred_indices, gt_indices = linear_sum_assignment(costs)
    matches = {}
    matched_ious = []
    for pred_index, gt_index in zip(pred_indices, gt_indices):
        iou = 1.0 - float(costs[pred_index, gt_index])
        if iou >= iou_threshold:
            matches[pred_index] = gt_index
            matched_ious.append(iou)
    tp = len(matches)
    return (matches, tp, len(predictions) - tp, len(ground_truth) - tp,
            float(np.mean(matched_ious)) if matched_ious else 0.0)


def raw_pair_metrics(pair_path: Path, gt_frames: dict[int, list[dict]],
                     score_threshold: float, iou_threshold: float) -> dict[str, float]:
    records = read_pair_detections(pair_path)
    pairs = defaultdict(list)
    for record in records:
        pairs[(record['prev_frame'], record['curr_frame'])].append(record)

    frame_predictions = defaultdict(list)
    for (prev_frame, curr_frame), detections in pairs.items():
        if prev_frame not in frame_predictions:
            frame_predictions[prev_frame] = [{
                'poly': det['prev_poly'], 'score': det['prev_score'],
                'cls': det['prev_cls']
            } for det in detections if det['prev_score'] >= score_threshold]
        frame_predictions[curr_frame] = [{
            'poly': det['curr_poly'], 'score': det['curr_score'],
            'cls': det['curr_cls']
        } for det in detections if det['curr_score'] >= score_threshold]

    tp = fp = fn = 0
    matched_iou_sum = 0.0
    for frame in sorted(set(gt_frames) | set(frame_predictions)):
        _, frame_tp, frame_fp, frame_fn, mean_iou = match_predictions(
            frame_predictions.get(frame, []), gt_frames.get(frame, []),
            iou_threshold)
        tp += frame_tp
        fp += frame_fp
        fn += frame_fn
        matched_iou_sum += frame_tp * mean_iou

    correct_links = wrong_links = one_side_links = 0
    transition_total = 0
    for (prev_frame, curr_frame), detections in pairs.items():
        candidates = [det for det in detections
                      if det['prev_score'] >= score_threshold
                      and det['curr_score'] >= score_threshold]
        prev_predictions = [{'poly': det['prev_poly']} for det in candidates]
        curr_predictions = [{'poly': det['curr_poly']} for det in candidates]
        prev_gt = gt_frames.get(prev_frame, [])
        curr_gt = gt_frames.get(curr_frame, [])
        prev_matches, _, _, _, _ = match_predictions(
            prev_predictions, prev_gt, iou_threshold)
        curr_matches, _, _, _, _ = match_predictions(
            curr_predictions, curr_gt, iou_threshold)
        prev_ids = {target['id'] for target in prev_gt}
        curr_ids = {target['id'] for target in curr_gt}
        transition_total += len(prev_ids & curr_ids)
        for index in range(len(candidates)):
            prev_match = prev_matches.get(index)
            curr_match = curr_matches.get(index)
            if prev_match is None or curr_match is None:
                one_side_links += 1
            elif prev_gt[prev_match]['id'] == curr_gt[curr_match]['id']:
                correct_links += 1
            else:
                wrong_links += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        'raw_tp': tp,
        'raw_fp': fp,
        'raw_fn': fn,
        'raw_precision': precision * 100,
        'raw_recall': recall * 100,
        'raw_f1': f1 * 100,
        'raw_loc_iou': matched_iou_sum / max(tp, 1) * 100,
        'pair_correct': correct_links,
        'pair_wrong': wrong_links,
        'pair_one_side': one_side_links,
        'pair_transition_total': transition_total,
        'pair_precision': correct_links / max(correct_links + wrong_links, 1) * 100,
        'pair_recall': correct_links / max(transition_total, 1) * 100,
    }


def class_summary(path: Path) -> dict[str, dict[str, float]]:
    result = {}
    for row in read_csv(path):
        if row['cls'] in CLASS_NAMES:
            result[row['cls']] = {
                key: float(row[key]) for key in
                ('HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'MOTA', 'IDF1',
                 'IDSW', 'Frag', 'CLR_FP', 'CLR_FN', 'GT_Dets')
            }
    return result


def sequence_class_rows(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    result = defaultdict(dict)
    for row in read_csv(path):
        result[row['seq']][row['cls']] = {
            key: float(row[key]) for key in
            ('HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1', 'IDSW', 'Frag',
             'CLR_TP', 'CLR_FN', 'CLR_FP', 'IDTP', 'IDFN', 'IDFP',
             'GT_Dets')
        }
    return result


def combined_sequence_metrics(
        eval_dir: Path,
        seq_classes: dict[str, dict[str, dict[str, float]]]
        ) -> dict[str, dict[str, float]]:
    """Rebuild TrackEval's per-sequence detection-averaged combination."""
    alpha_suffixes = [f'___{value}' for value in range(5, 100, 5)]
    detailed = {}
    for class_name in CLASS_NAMES:
        detailed[class_name] = {
            row['seq']: row for row in
            read_csv(eval_dir / f'{class_name}_detailed.csv')
            if row['seq'] != 'COMBINED'
        }

    result = {}
    for sequence, classes in seq_classes.items():
        tp = np.zeros(len(alpha_suffixes), dtype=np.float64)
        fn = np.zeros_like(tp)
        fp = np.zeros_like(tp)
        ass_weighted = np.zeros_like(tp)
        loc_weighted = np.zeros_like(tp)
        for class_name in CLASS_NAMES:
            row = detailed[class_name][sequence]
            class_tp = np.asarray([
                float(row[f'HOTA_TP{suffix}']) for suffix in alpha_suffixes])
            tp += class_tp
            fn += np.asarray([
                float(row[f'HOTA_FN{suffix}']) for suffix in alpha_suffixes])
            fp += np.asarray([
                float(row[f'HOTA_FP{suffix}']) for suffix in alpha_suffixes])
            ass_weighted += class_tp * np.asarray([
                float(row[f'AssA{suffix}']) for suffix in alpha_suffixes])
            loc_weighted += class_tp * np.asarray([
                float(row[f'LocA{suffix}']) for suffix in alpha_suffixes])
        ass_a = ass_weighted / np.maximum(tp, 1e-10)
        loc_a = loc_weighted / np.maximum(tp, 1e-10)
        det_a = tp / np.maximum(tp + fn + fp, 1.0)
        det_re = tp / np.maximum(tp + fn, 1.0)
        det_pr = tp / np.maximum(tp + fp, 1.0)
        hota = np.sqrt(det_a * ass_a)

        totals = {key: sum(classes[name][key] for name in CLASS_NAMES)
                  for key in ('IDSW', 'Frag', 'CLR_FP', 'CLR_FN', 'CLR_TP',
                              'IDTP', 'IDFN', 'IDFP', 'GT_Dets')}
        gt_dets = totals['CLR_TP'] + totals['CLR_FN']
        result[sequence] = {
            'HOTA': float(hota.mean() * 100),
            'DetA': float(det_a.mean() * 100),
            'AssA': float(ass_a.mean() * 100),
            'DetRe': float(det_re.mean() * 100),
            'DetPr': float(det_pr.mean() * 100),
            'LocA': float(loc_a.mean() * 100),
            'MOTA': ((totals['CLR_TP'] - totals['CLR_FP'] - totals['IDSW']) /
                     max(gt_dets, 1) * 100),
            'IDF1': (2 * totals['IDTP'] /
                     max(2 * totals['IDTP'] + totals['IDFN'] +
                         totals['IDFP'], 1) * 100),
            'IDSW': totals['IDSW'],
            'Frag': totals['Frag'],
            'CLR_FP': totals['CLR_FP'],
            'CLR_FN': totals['CLR_FN'],
            'GT_Dets': totals['GT_Dets'],
        }
    return result


def class_average_sequence_metrics(
        seq_classes: dict[str, dict[str, dict[str, float]]]
        ) -> dict[str, dict[str, float]]:
    result = {}
    for sequence, classes in seq_classes.items():
        result[sequence] = {
            metric: float(np.mean([
                classes[class_name][metric] for class_name in CLASS_NAMES]))
            for metric in ('HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1')
        }
    return result


def advantage(delta: float, experiment_name: str = '0719_02',
              base_name: str = 'BASE') -> str:
    if abs(delta) < 0.05:
        return 'TIE'
    return experiment_name if delta > 0 else base_name


def draw_polygon(image: np.ndarray, item: dict, color: tuple[int, int, int],
                 show_class: bool = False) -> None:
    points = np.round(item['poly']).astype(np.int32)
    cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
    center = tuple(points.mean(axis=0).astype(int))
    label = str(item['id'])
    if show_class:
        cls_index = item.get('cls', -1)
        cls_name = CLASS_NAMES[cls_index] if 0 <= cls_index < 8 else str(cls_index)
        label = f'{label}:{cls_name}'
    cv2.putText(image, label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                color, 1, cv2.LINE_AA)


def image_path(image_root: Path, sequence: str, frame: int) -> Path:
    preferred = image_root / sequence / f'{frame:06d}_p2.jpg'
    if preferred.exists():
        return preferred
    candidates = sorted((image_root / sequence).glob(f'{frame:06d}_p*.jpg'))
    if not candidates:
        raise FileNotFoundError(f'No visualization image for {sequence} frame {frame}')
    return candidates[0]


def choose_frames(gt: dict[int, list[dict]], base: dict[int, list[dict]],
                  experiment: dict[int, list[dict]], count: int = 3) -> list[int]:
    frames = sorted(gt)
    ranked = sorted(frames, key=lambda frame: (
        abs(len(base.get(frame, [])) - len(gt.get(frame, []))) +
        abs(len(experiment.get(frame, [])) - len(gt.get(frame, []))) +
        abs(len(base.get(frame, [])) - len(experiment.get(frame, [])))),
                    reverse=True)
    chosen = []
    min_gap = max(len(frames) // 10, 1)
    for frame in ranked:
        if all(abs(frame - previous) >= min_gap for previous in chosen):
            chosen.append(frame)
        if len(chosen) == count:
            break
    for frame in np.linspace(frames[0], frames[-1], count, dtype=int):
        if len(chosen) == count:
            break
        if frame not in chosen:
            chosen.append(int(frame))
    return sorted(chosen[:count])


def sequence_visualization(output: Path, sequence: str, frames: list[int],
                           image_root: Path, gt: dict[int, list[dict]],
                           base: dict[int, list[dict]],
                           experiment: dict[int, list[dict]],
                           cls_delta: float, det_delta: float,
                           det_a_delta: float, ass_a_delta: float) -> None:
    panel_width, panel_height = 480, 360
    title_height, row_label_width = 150, 115
    canvas = np.full((title_height + 3 * panel_height,
                      row_label_width + len(frames) * panel_width, 3),
                     245, dtype=np.uint8)
    title = (f'{sequence} | det winner: {advantage(det_delta)} '
             f'(0719_02-Base {det_delta:+.2f}) | cls winner: '
             f'{advantage(cls_delta)} ({cls_delta:+.2f})')
    subtitle = f'DetA {det_a_delta:+.2f} | AssA {ass_a_delta:+.2f}'
    cv2.putText(canvas, title, (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (50, 50, 50), 2, cv2.LINE_AA)
    rows = [('GT', gt, (0, 180, 255)),
            ('BASE', base, (30, 180, 30)),
            ('0719_02', experiment, (230, 100, 20))]
    for row_index, (name, tracks, color) in enumerate(rows):
        y0 = title_height + row_index * panel_height
        cv2.putText(canvas, name, (8, y0 + panel_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        for column, frame in enumerate(frames):
            image = cv2.imread(str(image_path(image_root, sequence, frame)))
            if image is None:
                raise RuntimeError(f'Cannot read image for {sequence} frame {frame}')
            for item in tracks.get(frame, []):
                draw_polygon(image, item, color, show_class=(name == 'GT'))
            image = cv2.resize(image, (panel_width, panel_height),
                               interpolation=cv2.INTER_AREA)
            x0 = row_label_width + column * panel_width
            canvas[y0:y0 + panel_height, x0:x0 + panel_width] = image
            cv2.putText(canvas, f'frame {frame}', (x0 + 8, y0 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                        3, cv2.LINE_AA)
            cv2.putText(canvas, f'frame {frame}', (x0 + 8, y0 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20),
                        1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def aggregate_raw(rows: dict[str, dict[str, float]]) -> dict[str, float]:
    totals = {key: sum(row[key] for row in rows.values()) for key in
              ('raw_tp', 'raw_fp', 'raw_fn', 'pair_correct', 'pair_wrong',
               'pair_transition_total')}
    precision = totals['raw_tp'] / max(totals['raw_tp'] + totals['raw_fp'], 1)
    recall = totals['raw_tp'] / max(totals['raw_tp'] + totals['raw_fn'], 1)
    return {
        **totals,
        'raw_precision': precision * 100,
        'raw_recall': recall * 100,
        'raw_f1': 200 * precision * recall / max(precision + recall, 1e-12),
        'pair_precision': totals['pair_correct'] /
        max(totals['pair_correct'] + totals['pair_wrong'], 1) * 100,
        'pair_recall': totals['pair_correct'] /
        max(totals['pair_transition_total'], 1) * 100,
    }


def write_delta_plot(path: Path, rows: list[dict],
                     experiment_name: str = '0719_02',
                     base_name: str = 'Base') -> None:
    ordered = sorted(rows, key=lambda row: row['det_HOTA_delta'])
    names = [row['sequence'] for row in ordered]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(16, 14), sharey=True)
    axes[0].barh(y, [row['det_HOTA_delta'] for row in ordered],
                 color=['#c44e52' if row['det_HOTA_delta'] < 0 else '#4c9f70'
                        for row in ordered])
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_title(
        f'det HOTA delta: {experiment_name} - {base_name}')
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[1].barh(y, [row['DetA_delta'] for row in ordered], label='DetA',
                 color='#4c72b0', alpha=0.8)
    axes[1].barh(y, [row['AssA_delta'] for row in ordered], label='AssA',
                 color='#dd8452', alpha=0.65)
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_title('Component deltas')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fmt(value: float) -> str:
    return f'{value:.3f}'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-eval', type=Path, required=True)
    parser.add_argument('--experiment-eval', type=Path, required=True)
    parser.add_argument('--base-det', type=Path, required=True)
    parser.add_argument('--experiment-det', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--score-threshold', type=float, default=0.2)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--base-name', default='BASE')
    parser.add_argument('--experiment-name', default='0719_02')
    parser.add_argument('--title')
    parser.add_argument('--protocol')
    parser.add_argument('--visualize-sequences', nargs='*', default=[])
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_tracker = args.base_eval.parent / 'preds'
    experiment_tracker = args.experiment_eval.parent / 'preds'
    base_classes = class_summary(args.base_eval / 'all_cls_summary.csv')
    experiment_classes = class_summary(
        args.experiment_eval / 'all_cls_summary.csv')
    base_seq_classes = sequence_class_rows(args.base_eval / 'all_seq_summary.csv')
    experiment_seq_classes = sequence_class_rows(
        args.experiment_eval / 'all_seq_summary.csv')
    base_det_seq = combined_sequence_metrics(args.base_eval, base_seq_classes)
    experiment_det_seq = combined_sequence_metrics(
        args.experiment_eval, experiment_seq_classes)
    base_cls_seq = class_average_sequence_metrics(base_seq_classes)
    experiment_cls_seq = class_average_sequence_metrics(
        experiment_seq_classes)
    sequences = sorted(base_det_seq)

    base_raw, experiment_raw = {}, {}
    sequence_rows = []
    for sequence in sequences:
        gt = read_mot(args.data_root / 'mot' / f'{sequence}.txt')
        base_tracks = read_mot(base_tracker / f'{sequence}.txt')
        experiment_tracks = read_mot(experiment_tracker / f'{sequence}.txt')
        base_raw[sequence] = raw_pair_metrics(
            args.base_det / f'{sequence}.txt', gt,
            args.score_threshold, args.iou_threshold)
        experiment_raw[sequence] = raw_pair_metrics(
            args.experiment_det / f'{sequence}.txt', gt,
            args.score_threshold, args.iou_threshold)
        row = {'sequence': sequence}
        for metric in ('HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'MOTA',
                       'IDF1', 'IDSW', 'Frag', 'CLR_FP', 'CLR_FN', 'GT_Dets'):
            row[f'base_{metric}'] = base_det_seq[sequence][metric]
            row[f'experiment_{metric}'] = experiment_det_seq[sequence][metric]
            row[f'{metric}_delta'] = (
                experiment_det_seq[sequence][metric] -
                base_det_seq[sequence][metric])
        row['det_HOTA_delta'] = row['HOTA_delta']
        row['cls_HOTA_delta'] = (
            experiment_cls_seq[sequence]['HOTA'] - base_cls_seq[sequence]['HOTA'])
        for metric in ('raw_precision', 'raw_recall', 'raw_f1', 'raw_loc_iou',
                       'pair_precision', 'pair_recall'):
            row[f'base_{metric}'] = base_raw[sequence][metric]
            row[f'experiment_{metric}'] = experiment_raw[sequence][metric]
            row[f'{metric}_delta'] = (
                experiment_raw[sequence][metric] - base_raw[sequence][metric])
        if row['DetA_delta'] < 0 and row['AssA_delta'] < 0:
            row['cause'] = ('association-dominant' if abs(row['AssA_delta']) >
                            abs(row['DetA_delta']) else 'detection-dominant')
        elif row['AssA_delta'] < 0:
            row['cause'] = 'association-only'
        elif row['DetA_delta'] < 0:
            row['cause'] = 'detection-only'
        else:
            row['cause'] = 'improved'
        sequence_rows.append(row)
        if sequence in args.visualize_sequences:
            frames = choose_frames(gt, base_tracks, experiment_tracks)
            sequence_visualization(
                output_dir / 'sequences' / f'{sequence}.jpg', sequence,
                frames, args.data_root / 'npy2jpg', gt, base_tracks,
                experiment_tracks, row['cls_HOTA_delta'],
                row['det_HOTA_delta'], row['DetA_delta'], row['AssA_delta'])

    with (output_dir / 'sequence_metrics.csv').open(
            'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sequence_rows[0]))
        writer.writeheader()
        writer.writerows(sequence_rows)
    write_delta_plot(
        output_dir / 'sequence_delta_overview.png', sequence_rows,
        args.experiment_name, args.base_name)

    base_raw_total = aggregate_raw(base_raw)
    experiment_raw_total = aggregate_raw(experiment_raw)
    combined_base = next(row for row in read_csv(
        args.base_eval / 'all_cls_summary.csv') if row['cls'] == 'cls_comb_det_av')
    combined_experiment = next(row for row in read_csv(
        args.experiment_eval / 'all_cls_summary.csv')
                               if row['cls'] == 'cls_comb_det_av')
    base_det_a = float(combined_base['DetA'])
    exp_det_a = float(combined_experiment['DetA'])
    base_ass_a = float(combined_base['AssA'])
    exp_ass_a = float(combined_experiment['AssA'])
    det_log_loss = -0.5 * math.log(exp_det_a / base_det_a)
    ass_log_loss = -0.5 * math.log(exp_ass_a / base_ass_a)
    total_log_loss = det_log_loss + ass_log_loss

    lines = [
        f'# {args.title or f"{args.experiment_name} vs {args.base_name}: Detection and Tracking Diagnosis"}', '',
        args.protocol or (
            f'Comparison protocol: {args.base_name} versus '
            f'{args.experiment_name}; tracker parameters are identical.'), '',
        '## Overall decomposition', '',
        f'| metric | {args.base_name} | {args.experiment_name} | delta |',
        '| --- | ---: | ---: | ---: |',
    ]
    for metric in ('HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'MOTA', 'IDF1',
                   'IDSW', 'Frag', 'CLR_FP', 'CLR_FN'):
        base_value = float(combined_base[metric])
        experiment_value = float(combined_experiment[metric])
        lines.append(f'| {metric} | {fmt(base_value)} | '
                     f'{fmt(experiment_value)} | '
                     f'{experiment_value - base_value:+.3f} |')
    lines.extend([
        '',
        f'In the exact identity `HOTA = sqrt(DetA * AssA)`, the negative '
        f'log-HOTA change is `{total_log_loss:.5f}`. DetA contributes '
        f'`{det_log_loss / total_log_loss * 100:.1f}%` and AssA contributes '
        f'`{ass_log_loss / total_log_loss * 100:.1f}%` of the loss.', '',
        '## Raw pair-detection audit', '',
        f'Fixed diagnostic thresholds: side score >= `{args.score_threshold}` '
        f'and rotated IoU >= `{args.iou_threshold}`. This audit is not the '
        'paper AP protocol; it isolates detector outputs before tracking.', '',
        f'| metric | {args.base_name} | {args.experiment_name} | delta |',
        '| --- | ---: | ---: | ---: |',
    ])
    for metric in ('raw_precision', 'raw_recall', 'raw_f1', 'pair_precision',
                   'pair_recall'):
        base_value = base_raw_total[metric]
        experiment_value = experiment_raw_total[metric]
        lines.append(f'| {metric} | {fmt(base_value)} | '
                     f'{fmt(experiment_value)} | '
                     f'{experiment_value - base_value:+.3f} |')
    lines.extend(['', '## Class comparison from all_cls_summary', '',
                  f'| class | {args.base_name} HOTA | {args.experiment_name} HOTA | delta | DetA delta | AssA delta |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for class_name in CLASS_NAMES:
        base = base_classes[class_name]
        experiment = experiment_classes[class_name]
        lines.append(
            f'| {class_name} | {fmt(base["HOTA"])} | '
            f'{fmt(experiment["HOTA"])} | '
            f'{experiment["HOTA"] - base["HOTA"]:+.3f} | '
            f'{experiment["DetA"] - base["DetA"]:+.3f} | '
            f'{experiment["AssA"] - base["AssA"]:+.3f} |')
    lines.extend(['', '## Sequence comparison from all_seq_summary', '',
                  'Rows are sorted by det HOTA delta.', '',
                  '| sequence | det winner | det HOTA delta | cls winner | cls HOTA delta | DetA delta | AssA delta | raw F1 delta | pair-link recall delta | diagnosis |',
                  '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |'])
    for row in sorted(sequence_rows, key=lambda item: item['det_HOTA_delta']):
        lines.append(
            f'| {row["sequence"]} | {advantage(row["det_HOTA_delta"], args.experiment_name, args.base_name)} | '
            f'{row["det_HOTA_delta"]:+.3f} | '
            f'{advantage(row["cls_HOTA_delta"], args.experiment_name, args.base_name)} | '
            f'{row["cls_HOTA_delta"]:+.3f} | {row["DetA_delta"]:+.3f} | '
            f'{row["AssA_delta"]:+.3f} | {row["raw_f1_delta"]:+.3f} | '
            f'{row["pair_recall_delta"]:+.3f} | {row["cause"]} |')
    lines.extend(['', '## Worst-sequence class breakdown', ''])
    for row in sorted(sequence_rows, key=lambda item: item['det_HOTA_delta'])[:5]:
        sequence = row['sequence']
        lines.extend([
            f'### {sequence}', '',
            f'- det HOTA `{row["det_HOTA_delta"]:+.3f}`; DetA '
            f'`{row["DetA_delta"]:+.3f}`; AssA `{row["AssA_delta"]:+.3f}`; '
            f'IDSW `{row["IDSW_delta"]:+.0f}`; Frag `{row["Frag_delta"]:+.0f}`.',
            f'- Raw detection F1 `{row["raw_f1_delta"]:+.3f}` and pair-link '
            f'recall `{row["pair_recall_delta"]:+.3f}`.',
            '',
            '| class | GT dets | HOTA delta | DetA delta | AssA delta |',
            '| --- | ---: | ---: | ---: | ---: |'])
        class_rows = []
        for class_name in CLASS_NAMES:
            base = base_seq_classes[sequence][class_name]
            experiment = experiment_seq_classes[sequence][class_name]
            class_rows.append((class_name, base['GT_Dets'],
                               experiment['HOTA'] - base['HOTA'],
                               experiment['DetA'] - base['DetA'],
                               experiment['AssA'] - base['AssA']))
        for class_name, gt_dets, hota, det_a, ass_a in sorted(
                class_rows, key=lambda item: item[2]):
            lines.append(f'| {class_name} | {gt_dets:.0f} | {hota:+.3f} | '
                         f'{det_a:+.3f} | {ass_a:+.3f} |')
        lines.extend(['', f'Visualization: `sequences/{sequence}.jpg`.', ''])
    lines.extend([
        '## Attribution rule', '',
        '- `raw F1` measures class-agnostic side-box quality before tracking.',
        '- `pair-link recall` measures whether both boxes of a pair detection '
        'recover the same persistent GT identity.',
        '- `DetA` measures detection quality after tracker filtering; `AssA`, '
        '`IDSW`, and `Frag` measure final trajectory association.',
        '- A stable/improved raw detector with lower AssA is tracking-stage '
        'association degradation. Lower raw F1 or pair-link recall indicates '
        'the pair detector supplies weaker boxes or correspondence.', '',
        'Overview: `sequence_delta_overview.png`. Per-sequence images: '
        '`sequences/*.jpg`. Machine-readable values: `sequence_metrics.csv`.',
    ])
    (output_dir / 'report.md').write_text('\n'.join(lines) + '\n',
                                                 encoding='utf-8')


if __name__ == '__main__':
    main()
