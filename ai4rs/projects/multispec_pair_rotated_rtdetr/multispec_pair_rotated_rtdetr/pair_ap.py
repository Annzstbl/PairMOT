"""AP evaluators used by single-frame and pair-frame validation.

The implementation intentionally evaluates all predictions, without a fixed
score threshold.  This makes AP measure ranking, duplicate detections and
localisation together, while keeping validation independent from the scale of
training losses.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Sequence

import numpy as np
import torch

from mmrotate.structures.bbox import qbox2rbox, rbbox_overlaps


IOU_THRS = tuple(np.arange(0.5, 0.96, 0.05).round(2).tolist())


def _profile_enabled() -> bool:
    return os.getenv('PAIR_AP_PROFILE', '').lower() in {'1', 'true', 'yes'}


def _profile_add(profile: dict | None, key: str, seconds: float) -> None:
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + seconds


def _profile_count(profile: dict | None, key: str, value: int = 1) -> None:
    if profile is not None:
        profile[key] = profile.get(key, 0) + value


def _profile_report(profile: dict) -> None:
    total = profile.get('total', 0.0)
    lines = [
        '[PAIR_AP_PROFILE] mAP aggregation step timings:',
        f'[PAIR_AP_PROFILE] total={total:.3f}s '
        f'samples={profile.get("samples", 0)} '
        f'classes={profile.get("classes", 0)} '
        f'detections={profile.get("detections", 0)} '
        f'gts={profile.get("gts", 0)} '
        f'rbbox_calls={profile.get("rbbox_calls", 0)}',
    ]
    for key in (
            'label_scan',
            'class_total',
            'class_filter',
            'rbbox_iou',
            'overlap_prepare',
            'sort_detections',
            'match_pair',
            'match_prev',
            'match_curr',
            'match_prefetch',
            'match_loop',
            'ap_integral',
            'summary',
    ):
        value = profile.get(key, 0.0)
        pct = (value / total * 100.0) if total > 0 else 0.0
        lines.append(f'[PAIR_AP_PROFILE] {key}: {value:.3f}s ({pct:.1f}%)')
    print('\n'.join(lines), flush=True)


def to_rbox_tensor(boxes) -> torch.Tensor:
    tensor = boxes.tensor if hasattr(boxes, 'tensor') else boxes
    return qbox2rbox(tensor) if tensor.size(-1) == 8 else tensor


def _field(data, key: str):
    return data[key] if isinstance(data, dict) else getattr(data, key)


def _has_field(data, key: str) -> bool:
    return key in data if isinstance(data, dict) else hasattr(data, key)


def _ap_from_tp_fp(tp: np.ndarray, fp: np.ndarray, num_gt: int) -> float:
    if num_gt == 0:
        return float('nan')
    recall = np.cumsum(tp) / num_gt
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1)
    # Standard area-under-precision-envelope AP (VOC/COCO-style ranking AP).
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _ap_from_tp_fp_batch(tp: np.ndarray, fp: np.ndarray,
                         num_gt: int) -> np.ndarray:
    """Compute AP for T IoU thresholds simultaneously.

    Args:
        tp: float32 array of shape (T, D) – 1 where detection is true-positive.
        fp: float32 array of shape (T, D) – 1 where detection is false-positive.
        num_gt: total ground-truth count across all samples.

    Returns:
        float64 array of shape (T,) with one AP value per threshold.
    """
    if num_gt == 0:
        return np.full(tp.shape[0], float('nan'), dtype=np.float64)
    T = tp.shape[0]
    cumtp = np.cumsum(tp, axis=1)
    recall    = cumtp / num_gt
    precision = cumtp / np.maximum(cumtp + np.cumsum(fp, axis=1), 1)
    z = np.zeros((T, 1), dtype=np.float64)
    recall    = np.concatenate([z, recall,    np.ones((T, 1))], axis=1)
    precision = np.concatenate([z, precision, z              ], axis=1)
    precision = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
    return np.sum((recall[:, 1:] - recall[:, :-1]) * precision[:, 1:], axis=1)


def _class_cache(samples: Sequence[dict], label: int, pair: bool):
    """Build score-sorted detections and per-sample IoU overlap matrices.

    Pair mode uses two full-matrix rbbox_overlaps calls per sample (prev and
    curr sides) rather than up to six separate mode-filtered subset calls,
    eliminating branching while keeping matrix sizes identical to the original.
    """
    gt_by_sample: dict  = {}
    detections:   List  = []
    overlaps:     dict  = {}

    for sample_id, sample in enumerate(samples):
        gt_indices   = torch.nonzero(sample['gt_labels'] == label,
                                     as_tuple=False).flatten()
        pred_indices = torch.nonzero(sample['pred_labels'] == label,
                                     as_tuple=False).flatten()
        gt_by_sample[sample_id] = gt_indices.tolist()

        np_ = len(pred_indices)
        ng  = len(gt_indices)

        if not np_ or not ng:
            overlaps[sample_id] = np.empty((np_, ng), dtype=np.float32)
        elif pair:
            # Gather visibility flags for selected preds / GTs.
            pvp = sample['pred_valid_prev'][pred_indices].bool()  # (np_,)
            pvc = sample['pred_valid_curr'][pred_indices].bool()
            gvp = sample['gt_valid_prev'][gt_indices].bool()      # (ng,)
            gvc = sample['gt_valid_curr'][gt_indices].bool()

            # Compatibility: pred and GT must have the same visibility mode.
            compat = ((pvp[:, None] == gvp[None, :]) &
                      (pvc[:, None] == gvc[None, :]))             # (np_, ng)
            any_prev = pvp[:, None] & gvp[None, :]
            any_curr = pvc[:, None] & gvc[None, :]

            # Two vectorised rbbox_overlaps calls replace the original loop
            # over three visibility-mode subsets (up to six calls total).
            iou_p = rbbox_overlaps(sample['pred_prev'][pred_indices],
                                   sample['gt_prev'][gt_indices])  # (np_, ng)
            iou_c = rbbox_overlaps(sample['pred_curr'][pred_indices],
                                   sample['gt_curr'][gt_indices])

            # both-visible → min(iou_prev, iou_curr); single-side → that side.
            ov = torch.where(any_prev & any_curr,
                             torch.minimum(iou_p, iou_c),
                             torch.where(any_prev, iou_p, iou_c))
            ov = torch.where(compat & (any_prev | any_curr), ov,
                             ov.new_full((), -1.))
            overlaps[sample_id] = ov.cpu().numpy().astype(np.float32)
        else:
            overlaps[sample_id] = rbbox_overlaps(
                sample['pred_boxes'][pred_indices],
                sample['gt_boxes'][gt_indices]).cpu().numpy().astype(np.float32)

        for pred_idx in pred_indices.tolist():
            detections.append((float(sample['pred_scores'][pred_idx]),
                               sample_id, pred_idx))

    num_gt = sum(len(g) for g in gt_by_sample.values())
    detections.sort(key=lambda x: x[0], reverse=True)
    pred_row = {
        sid: {pi: r for r, pi in enumerate(
            torch.nonzero(samples[sid]['pred_labels'] == label,
                          as_tuple=False).flatten().tolist())}
        for sid in range(len(samples))}
    return num_gt, gt_by_sample, detections, overlaps, pred_row


def _class_aps(samples: Sequence[dict], label: int, pair: bool) -> List[float]:
    """AP at all IoU thresholds for one class.

    A single pass over score-sorted detections assigns TP/FP for all ten IoU
    thresholds simultaneously via numpy broadcasting, replacing the previous
    triple Python loop (thresholds × detections × GT-indices).
    """
    num_gt, gt_by_sample, detections, overlaps, pred_row = _class_cache(
        samples, label, pair)
    if num_gt == 0:
        return [float('nan')] * len(IOU_THRS)
    if not detections:
        return [0.0] * len(IOU_THRS)
    return _aps_from_cached_matches(num_gt, gt_by_sample, detections, overlaps,
                                    pred_row)


def _aps_from_cached_matches(num_gt: int,
                             gt_by_sample: Dict[int, List[int]],
                             detections: List[tuple],
                             overlaps: Dict[int, np.ndarray],
                             pred_row: Dict[int, Dict[int, int]],
                             profile: dict | None = None,
                             profile_key: str = 'match') -> List[float]:
    """Compute AP from prepared detections and overlap rows."""
    t_total = time.perf_counter()
    if num_gt == 0:
        return [float('nan')] * len(IOU_THRS)
    if not detections:
        return [0.0] * len(IOU_THRS)

    T   = len(IOU_THRS)
    thr = np.array(IOU_THRS, dtype=np.float32)  # (T,)
    D   = len(detections)

    # Initialise as all-FP; flip to TP where matched.
    tp = np.zeros((T, D), dtype=np.float32)
    fp = np.ones( (T, D), dtype=np.float32)

    # Per-sample matched-GT bookkeeping: shape (T, num_gt_in_sample).
    matched = {
        sid: np.zeros((T, len(gts)), dtype=bool)
        for sid, gts in gt_by_sample.items() if gts}

    # Pre-fetch overlap rows once to avoid per-iteration dict look-ups.
    t0 = time.perf_counter()
    det_meta = [
        (sid, overlaps[sid][pred_row[sid][pi]])
        for _, sid, pi in detections]
    _profile_add(profile, 'match_prefetch', time.perf_counter() - t0)

    arange_T = np.arange(T)
    t0 = time.perf_counter()
    for det_id, (sid, row) in enumerate(det_meta):
        if not gt_by_sample[sid]:
            continue  # no GT for this class in this sample → FP (already set)

        m   = matched[sid]                           # (T, ng) bool
        # row (ng,) broadcast to (T, ng); mask already-matched columns with -1
        eff = np.where(m, np.float32(-1.), row)      # (T, ng)

        best_col = np.argmax(eff, axis=1)            # (T,)
        best_iou = eff[arange_T, best_col]           # (T,)

        is_tp = best_iou >= thr                      # (T,) bool
        tp[is_tp, det_id] = 1.
        fp[is_tp, det_id] = 0.

        tp_idx = np.nonzero(is_tp)[0]
        if tp_idx.size:
            # Vectorised update: mark matched GT for every TP threshold at once.
            m[tp_idx, best_col[tp_idx]] = True
    _profile_add(profile, 'match_loop', time.perf_counter() - t0)

    t0 = time.perf_counter()
    aps = _ap_from_tp_fp_batch(tp, fp, num_gt).tolist()
    _profile_add(profile, 'ap_integral', time.perf_counter() - t0)
    _profile_add(profile, profile_key, time.perf_counter() - t_total)
    return aps


def _class_multi_aps(samples: Sequence[dict],
                     label: int,
                     profile: dict | None = None) -> Dict[str, List[float]]:
    """Compute pair, prev-independent and curr-independent AP for one class.

    The previous implementation called ``_summary`` three times, which rebuilt
    score lists and rotated IoU matrices for pair, prev and curr independently.
    Here the expensive prev/curr IoU matrices are built once per class/sample
    and reused by all three ranking evaluations.
    """
    gt_by_sample_pair: Dict[int, List[int]] = {}
    gt_by_sample_prev: Dict[int, List[int]] = {}
    gt_by_sample_curr: Dict[int, List[int]] = {}
    pair_overlaps: Dict[int, np.ndarray] = {}
    prev_overlaps: Dict[int, np.ndarray] = {}
    curr_overlaps: Dict[int, np.ndarray] = {}
    pred_row: Dict[int, Dict[int, int]] = {}
    pair_detections: List[tuple] = []
    prev_detections: List[tuple] = []
    curr_detections: List[tuple] = []

    t_class = time.perf_counter()
    for sample_id, sample in enumerate(samples):
        t0 = time.perf_counter()
        gt_indices = torch.nonzero(
            sample['gt_labels'] == label, as_tuple=False).flatten()
        pred_indices = torch.nonzero(
            sample['pred_labels'] == label, as_tuple=False).flatten()
        gt_by_sample_pair[sample_id] = gt_indices.tolist()

        pred_row[sample_id] = {
            int(pred_idx): row
            for row, pred_idx in enumerate(pred_indices.tolist())
        }
        for pred_idx in pred_indices.tolist():
            pair_detections.append(
                (float(sample['pred_scores'][pred_idx]), sample_id, pred_idx))
            prev_detections.append(
                (float(sample['pred_score_prev'][pred_idx]), sample_id,
                 pred_idx))
            curr_detections.append(
                (float(sample['pred_score_curr'][pred_idx]), sample_id,
                 pred_idx))

        if len(gt_indices):
            valid_prev = sample['gt_valid_prev'][gt_indices].bool()
            valid_curr = sample['gt_valid_curr'][gt_indices].bool()
            prev_cols = torch.nonzero(valid_prev, as_tuple=False).flatten()
            curr_cols = torch.nonzero(valid_curr, as_tuple=False).flatten()
            gt_by_sample_prev[sample_id] = gt_indices[prev_cols].tolist()
            gt_by_sample_curr[sample_id] = gt_indices[curr_cols].tolist()
        else:
            valid_prev = torch.empty(0, dtype=torch.bool)
            valid_curr = torch.empty(0, dtype=torch.bool)
            prev_cols = torch.empty(0, dtype=torch.long)
            curr_cols = torch.empty(0, dtype=torch.long)
            gt_by_sample_prev[sample_id] = []
            gt_by_sample_curr[sample_id] = []
        _profile_add(profile, 'class_filter', time.perf_counter() - t0)
        _profile_count(profile, 'detections', len(pred_indices))
        _profile_count(profile, 'gts', len(gt_indices))

        np_ = len(pred_indices)
        ng = len(gt_indices)
        if not np_ or not ng:
            pair_overlaps[sample_id] = np.empty((np_, ng), dtype=np.float32)
            prev_overlaps[sample_id] = np.empty(
                (np_, len(prev_cols)), dtype=np.float32)
            curr_overlaps[sample_id] = np.empty(
                (np_, len(curr_cols)), dtype=np.float32)
            continue

        t0 = time.perf_counter()
        iou_prev = rbbox_overlaps(sample['pred_prev'][pred_indices],
                                  sample['gt_prev'][gt_indices])
        iou_curr = rbbox_overlaps(sample['pred_curr'][pred_indices],
                                  sample['gt_curr'][gt_indices])
        _profile_add(profile, 'rbbox_iou', time.perf_counter() - t0)
        _profile_count(profile, 'rbbox_calls', 2)

        t0 = time.perf_counter()
        pvp = sample['pred_valid_prev'][pred_indices].bool()
        pvc = sample['pred_valid_curr'][pred_indices].bool()
        gvp = valid_prev
        gvc = valid_curr
        compat = ((pvp[:, None] == gvp[None, :]) &
                  (pvc[:, None] == gvc[None, :]))
        any_prev = pvp[:, None] & gvp[None, :]
        any_curr = pvc[:, None] & gvc[None, :]
        pair_overlap = torch.where(
            any_prev & any_curr,
            torch.minimum(iou_prev, iou_curr),
            torch.where(any_prev, iou_prev, iou_curr))
        pair_overlap = torch.where(
            compat & (any_prev | any_curr), pair_overlap,
            pair_overlap.new_full((), -1.))
        pair_overlaps[sample_id] = pair_overlap.cpu().numpy().astype(
            np.float32)
        prev_overlaps[sample_id] = iou_prev[:, prev_cols].cpu().numpy().astype(
            np.float32)
        curr_overlaps[sample_id] = iou_curr[:, curr_cols].cpu().numpy().astype(
            np.float32)
        _profile_add(profile, 'overlap_prepare', time.perf_counter() - t0)

    t0 = time.perf_counter()
    pair_detections.sort(key=lambda x: x[0], reverse=True)
    prev_detections.sort(key=lambda x: x[0], reverse=True)
    curr_detections.sort(key=lambda x: x[0], reverse=True)
    _profile_add(profile, 'sort_detections', time.perf_counter() - t0)
    result = dict(
        pair=_aps_from_cached_matches(
            sum(len(gts) for gts in gt_by_sample_pair.values()),
            gt_by_sample_pair, pair_detections, pair_overlaps, pred_row,
            profile, 'match_pair'),
        prev=_aps_from_cached_matches(
            sum(len(gts) for gts in gt_by_sample_prev.values()),
            gt_by_sample_prev, prev_detections, prev_overlaps, pred_row,
            profile, 'match_prev'),
        curr=_aps_from_cached_matches(
            sum(len(gts) for gts in gt_by_sample_curr.values()),
            gt_by_sample_curr, curr_detections, curr_overlaps, pred_row,
            profile, 'match_curr'),
    )
    _profile_add(profile, 'class_total', time.perf_counter() - t_class)
    return result


def _gt_filter_cols(valid_prev: torch.Tensor, valid_curr: torch.Tensor,
                    mode: str) -> torch.Tensor:
    if mode == 'all':
        keep = torch.ones_like(valid_prev, dtype=torch.bool)
    elif mode == 'both':
        keep = valid_prev & valid_curr
    elif mode == 'new':
        keep = (~valid_prev) & valid_curr
    elif mode == 'disappear':
        keep = valid_prev & (~valid_curr)
    else:
        raise ValueError(f'Unsupported GT filter mode: {mode}')
    return torch.nonzero(keep, as_tuple=False).flatten()


def _class_multi_aps_by_gt_filter(
        samples: Sequence[dict],
        label: int,
        gt_filters: Sequence[str],
        profile: dict | None = None) -> Dict[str, Dict[str, List[float]]]:
    """Compute AP for several GT filters while sharing IoU matrices.

    The expensive part is rotated IoU between predictions and all union GTs.
    Build it once for each class/sample, then select GT columns for
    all/both/new/disappear AP views.
    """
    filters = tuple(dict.fromkeys(gt_filters))
    gt_by_filter = {
        mode: {
            'pair': {},
            'prev': {},
            'curr': {},
        } for mode in filters
    }
    overlaps_by_filter = {
        mode: {
            'pair': {},
            'prev': {},
            'curr': {},
        } for mode in filters
    }
    pred_row: Dict[int, Dict[int, int]] = {}
    pair_detections: List[tuple] = []
    prev_detections: List[tuple] = []
    curr_detections: List[tuple] = []

    t_class = time.perf_counter()
    for sample_id, sample in enumerate(samples):
        t0 = time.perf_counter()
        gt_indices = torch.nonzero(
            sample['gt_labels'] == label, as_tuple=False).flatten()
        pred_indices = torch.nonzero(
            sample['pred_labels'] == label, as_tuple=False).flatten()
        pred_row[sample_id] = {
            int(pred_idx): row
            for row, pred_idx in enumerate(pred_indices.tolist())
        }
        for pred_idx in pred_indices.tolist():
            pair_detections.append(
                (float(sample['pred_scores'][pred_idx]), sample_id, pred_idx))
            prev_detections.append(
                (float(sample['pred_score_prev'][pred_idx]), sample_id,
                 pred_idx))
            curr_detections.append(
                (float(sample['pred_score_curr'][pred_idx]), sample_id,
                 pred_idx))

        if len(gt_indices):
            valid_prev = sample['gt_valid_prev'][gt_indices].bool()
            valid_curr = sample['gt_valid_curr'][gt_indices].bool()
        else:
            valid_prev = torch.empty(0, dtype=torch.bool)
            valid_curr = torch.empty(0, dtype=torch.bool)
        filter_cols = {
            mode: _gt_filter_cols(valid_prev, valid_curr, mode)
            for mode in filters
        }
        _profile_add(profile, 'class_filter', time.perf_counter() - t0)
        _profile_count(profile, 'detections', len(pred_indices))
        _profile_count(profile, 'gts', len(gt_indices))

        np_ = len(pred_indices)
        ng = len(gt_indices)
        if np_ and ng:
            t0 = time.perf_counter()
            iou_prev = rbbox_overlaps(sample['pred_prev'][pred_indices],
                                      sample['gt_prev'][gt_indices])
            iou_curr = rbbox_overlaps(sample['pred_curr'][pred_indices],
                                      sample['gt_curr'][gt_indices])
            _profile_add(profile, 'rbbox_iou', time.perf_counter() - t0)
            _profile_count(profile, 'rbbox_calls', 2)

            t0 = time.perf_counter()
            pvp = sample['pred_valid_prev'][pred_indices].bool()
            pvc = sample['pred_valid_curr'][pred_indices].bool()
            compat = ((pvp[:, None] == valid_prev[None, :]) &
                      (pvc[:, None] == valid_curr[None, :]))
            any_prev = pvp[:, None] & valid_prev[None, :]
            any_curr = pvc[:, None] & valid_curr[None, :]
            pair_overlap = torch.where(
                any_prev & any_curr,
                torch.minimum(iou_prev, iou_curr),
                torch.where(any_prev, iou_prev, iou_curr))
            pair_overlap = torch.where(
                compat & (any_prev | any_curr), pair_overlap,
                pair_overlap.new_full((), -1.))
            _profile_add(profile, 'overlap_prepare', time.perf_counter() - t0)
        else:
            iou_prev = torch.empty((np_, ng), dtype=torch.float32)
            iou_curr = torch.empty((np_, ng), dtype=torch.float32)
            pair_overlap = torch.empty((np_, ng), dtype=torch.float32)

        for mode, cols in filter_cols.items():
            gt_by_filter[mode]['pair'][sample_id] = gt_indices[cols].tolist()
            prev_cols = cols[valid_prev[cols]] if cols.numel() else cols
            curr_cols = cols[valid_curr[cols]] if cols.numel() else cols
            gt_by_filter[mode]['prev'][sample_id] = gt_indices[prev_cols].tolist()
            gt_by_filter[mode]['curr'][sample_id] = gt_indices[curr_cols].tolist()
            overlaps_by_filter[mode]['pair'][sample_id] = (
                pair_overlap[:, cols].cpu().numpy().astype(np.float32))
            overlaps_by_filter[mode]['prev'][sample_id] = (
                iou_prev[:, prev_cols].cpu().numpy().astype(np.float32))
            overlaps_by_filter[mode]['curr'][sample_id] = (
                iou_curr[:, curr_cols].cpu().numpy().astype(np.float32))

    t0 = time.perf_counter()
    pair_detections.sort(key=lambda x: x[0], reverse=True)
    prev_detections.sort(key=lambda x: x[0], reverse=True)
    curr_detections.sort(key=lambda x: x[0], reverse=True)
    _profile_add(profile, 'sort_detections', time.perf_counter() - t0)

    result = {}
    for mode in filters:
        result[mode] = dict(
            pair=_aps_from_cached_matches(
                sum(len(gts) for gts in gt_by_filter[mode]['pair'].values()),
                gt_by_filter[mode]['pair'], pair_detections,
                overlaps_by_filter[mode]['pair'], pred_row, profile,
                f'match_{mode}_pair'),
            prev=_aps_from_cached_matches(
                sum(len(gts) for gts in gt_by_filter[mode]['prev'].values()),
                gt_by_filter[mode]['prev'], prev_detections,
                overlaps_by_filter[mode]['prev'], pred_row, profile,
                f'match_{mode}_prev'),
            curr=_aps_from_cached_matches(
                sum(len(gts) for gts in gt_by_filter[mode]['curr'].values()),
                gt_by_filter[mode]['curr'], curr_detections,
                overlaps_by_filter[mode]['curr'], pred_row, profile,
                f'match_{mode}_curr'),
        )
    _profile_add(profile, 'class_total', time.perf_counter() - t_class)
    return result


def _summary(samples: Sequence[dict], pair: bool, prefix: str) -> Dict[str, float]:
    labels = [sample['gt_labels'] for sample in samples]
    if not labels or not any(label.numel() for label in labels):
        return {f'{prefix}_AP50': 0.0, f'{prefix}_AP75': 0.0,
                f'{prefix}_mAP50_95': 0.0}
    num_classes = int(torch.cat(labels).max().item()) + 1
    per_thr = []
    metrics = {}
    class_ap_grid = [_class_aps(samples, label, pair)
                     for label in range(num_classes)]
    for label, aps in enumerate(class_ap_grid):
        if not np.isnan(aps[0]):
            metrics[f'{prefix}_class{label}_AP50'] = float(aps[0])
    for thr_idx, thr in enumerate(IOU_THRS):
        class_aps = [aps[thr_idx] for aps in class_ap_grid]
        valid = [ap for ap in class_aps if not np.isnan(ap)]
        value = float(np.mean(valid)) if valid else 0.0
        per_thr.append(value)
        metrics[f'{prefix}_AP{int(thr * 100):02d}'] = value
    metrics[f'{prefix}_mAP50_95'] = float(np.mean(per_thr))
    return metrics


def serialize_pair_sample(gt,
                          pred,
                          pres_thr: float = 0.5,
                          max_dets: int | None = 100) -> dict:
    """Move the pair fields needed for AP to CPU and derive pair scores."""
    gt_labels = _field(gt, 'labels').detach().cpu().long()
    pred_labels = _field(pred, 'labels').detach().cpu().long()
    cls_scores = _field(pred, 'scores').detach().cpu().float().clamp(0, 1)
    if _has_field(pred, 'scores_prev'):
        score_prev = _field(pred, 'scores_prev').detach().cpu().float().clamp(0, 1)
        score_curr = _field(pred, 'scores_curr').detach().cpu().float().clamp(0, 1)
        pred_valid_prev = score_prev >= pres_thr
        pred_valid_curr = score_curr >= pres_thr
        pair_scores = torch.where(
            pred_valid_prev & pred_valid_curr,
            torch.sqrt(score_prev.clamp(min=1e-6) *
                       score_curr.clamp(min=1e-6)),
            torch.where(
                pred_valid_prev,
                score_prev * (1 - score_curr),
                (1 - score_prev) * score_curr))
        valid_pair = pred_valid_prev | pred_valid_curr
        pred_labels = pred_labels[valid_pair]
        pred_valid_prev = pred_valid_prev[valid_pair]
        pred_valid_curr = pred_valid_curr[valid_pair]
        pair_scores = pair_scores[valid_pair]
        cls_scores = cls_scores[valid_pair]
        score_prev = score_prev[valid_pair]
        score_curr = score_curr[valid_pair]
        pred_prev_all = to_rbox_tensor(
            _field(pred, 'bboxes_prev')).detach().cpu().float()
        pred_curr_all = to_rbox_tensor(
            _field(pred, 'bboxes_curr')).detach().cpu().float()
        pred_prev = pred_prev_all[valid_pair]
        pred_curr = pred_curr_all[valid_pair]
    else:
        pres_prev = _field(pred, 'presence_prev').detach().cpu().float().clamp(0, 1)
        pres_curr = _field(pred, 'presence_curr').detach().cpu().float().clamp(0, 1)
        score_prev = cls_scores * pres_prev
        score_curr = cls_scores * pres_curr
        pred_valid_prev = pres_prev >= pres_thr
        pred_valid_curr = pres_curr >= pres_thr
        # Score the visibility mode that the query explicitly predicts.
        pair_scores = cls_scores * torch.where(
            pred_valid_prev & pred_valid_curr, torch.sqrt(pres_prev * pres_curr),
            torch.where(pred_valid_prev, pres_prev * (1 - pres_curr),
                        (1 - pres_prev) * pres_curr))
        pred_prev = to_rbox_tensor(_field(pred, 'bboxes_prev')).detach().cpu().float()
        pred_curr = to_rbox_tensor(_field(pred, 'bboxes_curr')).detach().cpu().float()
    if max_dets is not None and pair_scores.numel() > max_dets:
        independent_scores = torch.maximum(score_prev, score_curr)
        rank_scores = torch.maximum(pair_scores, independent_scores)
        keep = torch.topk(rank_scores, k=max_dets).indices
        pred_labels = pred_labels[keep]
        pred_valid_prev = pred_valid_prev[keep]
        pred_valid_curr = pred_valid_curr[keep]
        pair_scores = pair_scores[keep]
        cls_scores = cls_scores[keep]
        score_prev = score_prev[keep]
        score_curr = score_curr[keep]
        pred_prev = pred_prev[keep]
        pred_curr = pred_curr[keep]
    return dict(
        gt_labels=gt_labels,
        gt_prev=to_rbox_tensor(_field(gt, 'bboxes_prev')).detach().cpu().float(),
        gt_curr=to_rbox_tensor(_field(gt, 'bboxes_curr')).detach().cpu().float(),
        gt_valid_prev=_field(gt, 'valid_prev').detach().cpu().bool(),
        gt_valid_curr=_field(gt, 'valid_curr').detach().cpu().bool(),
        pred_labels=pred_labels,
        pred_prev=pred_prev,
        pred_curr=pred_curr,
        pred_valid_prev=pred_valid_prev,
        pred_valid_curr=pred_valid_curr,
        pred_scores=pair_scores,
        pred_cls_scores=cls_scores,
        pred_presence_prev=score_prev,
        pred_presence_curr=score_curr,
        pred_score_prev=score_prev,
        pred_score_curr=score_curr)


def pair_ap_metrics(samples: Sequence[dict]) -> Dict[str, float]:
    return _summary(samples, pair=True, prefix='pair')


def pair_and_independent_ap_metrics(
        pair_samples: Sequence[dict]) -> Dict[str, float]:
    """Compute pair AP and independent AP while sharing IoU caches."""
    profile = {'samples': len(pair_samples)} if _profile_enabled() else None
    t_total = time.perf_counter()
    t0 = time.perf_counter()
    labels = [sample['gt_labels'] for sample in pair_samples]
    _profile_add(profile, 'label_scan', time.perf_counter() - t0)
    if not labels or not any(label.numel() for label in labels):
        metrics = {
            'pair_AP50': 0.0,
            'pair_AP75': 0.0,
            'pair_mAP50_95': 0.0,
            'independent_prev_AP50': 0.0,
            'independent_prev_AP75': 0.0,
            'independent_prev_mAP50_95': 0.0,
            'independent_curr_AP50': 0.0,
            'independent_curr_AP75': 0.0,
            'independent_curr_mAP50_95': 0.0,
            'independent_AP50': 0.0,
            'independent_AP75': 0.0,
            'independent_mAP50_95': 0.0,
            'association_gap_AP50': 0.0,
        }
        return metrics

    num_classes = int(torch.cat(labels).max().item()) + 1
    _profile_count(profile, 'classes', num_classes)
    per_class = [_class_multi_aps(pair_samples, label, profile)
                 for label in range(num_classes)]
    metrics: Dict[str, float] = {}

    def add_summary(name: str, mode: str) -> None:
        per_thr = []
        for label, values_by_mode in enumerate(per_class):
            aps = values_by_mode[mode]
            if not np.isnan(aps[0]):
                metrics[f'{name}_class{label}_AP50'] = float(aps[0])
        for thr_idx, thr in enumerate(IOU_THRS):
            class_aps = [values_by_mode[mode][thr_idx]
                         for values_by_mode in per_class]
            valid = [ap for ap in class_aps if not np.isnan(ap)]
            value = float(np.mean(valid)) if valid else 0.0
            per_thr.append(value)
            metrics[f'{name}_AP{int(thr * 100):02d}'] = value
        metrics[f'{name}_mAP50_95'] = float(np.mean(per_thr))

    t0 = time.perf_counter()
    add_summary('pair', 'pair')
    add_summary('independent_prev', 'prev')
    add_summary('independent_curr', 'curr')
    for name in ('AP50', 'AP75', 'mAP50_95'):
        metrics[f'independent_{name}'] = float(np.mean([
            metrics[f'independent_prev_{name}'],
            metrics[f'independent_curr_{name}'],
        ]))
    metrics['association_gap_AP50'] = (
        metrics['independent_AP50'] - metrics['pair_AP50'])
    _profile_add(profile, 'summary', time.perf_counter() - t0)
    if profile is not None:
        profile['total'] = time.perf_counter() - t_total
        _profile_report(profile)
    return metrics


def _add_pair_independent_summary(metrics: Dict[str, float],
                                  per_class: Sequence[Dict[str, List[float]]],
                                  prefix: str = '') -> None:
    def metric_name(name: str) -> str:
        return f'{prefix}_{name}' if prefix else name

    def add_summary(name: str, mode: str) -> None:
        out_name = metric_name(name)
        per_thr = []
        for label, values_by_mode in enumerate(per_class):
            aps = values_by_mode[mode]
            if not np.isnan(aps[0]):
                metrics[f'{out_name}_class{label}_AP50'] = float(aps[0])
        for thr_idx, thr in enumerate(IOU_THRS):
            class_aps = [values_by_mode[mode][thr_idx]
                         for values_by_mode in per_class]
            valid = [ap for ap in class_aps if not np.isnan(ap)]
            value = float(np.mean(valid)) if valid else 0.0
            per_thr.append(value)
            metrics[f'{out_name}_AP{int(thr * 100):02d}'] = value
        metrics[f'{out_name}_mAP50_95'] = float(np.mean(per_thr))

    add_summary('pair', 'pair')
    add_summary('independent_prev', 'prev')
    add_summary('independent_curr', 'curr')
    for name in ('AP50', 'AP75', 'mAP50_95'):
        metrics[metric_name(f'independent_{name}')] = float(np.mean([
            metrics[metric_name(f'independent_prev_{name}')],
            metrics[metric_name(f'independent_curr_{name}')],
        ]))
    metrics[metric_name('association_gap_AP50')] = (
        metrics[metric_name('independent_AP50')] -
        metrics[metric_name('pair_AP50')])


def pair_and_independent_ap_metrics_with_gt_filters(
        pair_samples: Sequence[dict],
        gt_filters: Sequence[str] = ('both', 'new', 'disappear')
) -> Dict[str, float]:
    """Compute all-GT AP plus filtered-GT AP with shared IoU caches."""
    filters = ('all', ) + tuple(
        mode for mode in dict.fromkeys(gt_filters) if mode != 'all')
    profile = {'samples': len(pair_samples)} if _profile_enabled() else None
    t_total = time.perf_counter()
    t0 = time.perf_counter()
    labels = [sample['gt_labels'] for sample in pair_samples]
    _profile_add(profile, 'label_scan', time.perf_counter() - t0)
    if not labels or not any(label.numel() for label in labels):
        empty = pair_and_independent_ap_metrics(pair_samples)
        metrics = dict(empty)
        for mode in filters:
            if mode == 'all':
                continue
            metrics.update({
                f'{mode}_{name}': value
                for name, value in empty.items()
            })
        return metrics

    num_classes = int(torch.cat(labels).max().item()) + 1
    _profile_count(profile, 'classes', num_classes)
    per_class_by_filter = [
        _class_multi_aps_by_gt_filter(pair_samples, label, filters, profile)
        for label in range(num_classes)
    ]
    metrics: Dict[str, float] = {}
    t0 = time.perf_counter()
    for mode in filters:
        per_class = [values_by_filter[mode]
                     for values_by_filter in per_class_by_filter]
        _add_pair_independent_summary(
            metrics, per_class, prefix='' if mode == 'all' else mode)
    _profile_add(profile, 'summary', time.perf_counter() - t0)
    if profile is not None:
        profile['total'] = time.perf_counter() - t_total
        _profile_report(profile)
    return metrics


def independent_ap_metrics(pair_samples: Sequence[dict],
                           pair_ap50: float | None = None) -> Dict[str, float]:
    metrics = {}
    side_metrics = []
    for side in ('prev', 'curr'):
        samples = []
        for sample in pair_samples:
            valid = sample[f'gt_valid_{side}']
            samples.append(dict(
                gt_labels=sample['gt_labels'][valid],
                gt_boxes=sample[f'gt_{side}'][valid],
                pred_labels=sample['pred_labels'],
                pred_boxes=sample[f'pred_{side}'],
                pred_scores=sample.get(
                    f'pred_score_{side}',
                    sample['pred_cls_scores'] *
                    sample[f'pred_presence_{side}'])))
        current = _summary(samples, pair=False, prefix=f'independent_{side}')
        metrics.update(current)
        side_metrics.append(current)
    for name in ('AP50', 'AP75', 'mAP50_95'):
        metrics[f'independent_{name}'] = float(np.mean([
            side_metrics[0][f'independent_prev_{name}'],
            side_metrics[1][f'independent_curr_{name}'],
        ]))
    if pair_ap50 is None:
        pair_ap50 = pair_ap_metrics(pair_samples)['pair_AP50']
    metrics['association_gap_AP50'] = metrics['independent_AP50'] - pair_ap50
    return metrics
