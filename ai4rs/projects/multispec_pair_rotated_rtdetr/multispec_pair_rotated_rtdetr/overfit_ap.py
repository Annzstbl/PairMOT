"""AP evaluators used by the single- and pair-frame overfit checks.

The implementation intentionally evaluates all predictions, without a fixed
score threshold.  This makes AP measure ranking, duplicate detections and
localisation together, while keeping the overfit acceptance independent from
the scale of training losses.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from mmrotate.structures.bbox import qbox2rbox, rbbox_overlaps


IOU_THRS = tuple(np.arange(0.5, 0.96, 0.05).round(2).tolist())


def to_rbox_tensor(boxes) -> torch.Tensor:
    tensor = boxes.tensor if hasattr(boxes, 'tensor') else boxes
    return qbox2rbox(tensor) if tensor.size(-1) == 8 else tensor


def _field(data, key: str):
    return data[key] if isinstance(data, dict) else getattr(data, key)


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


def _class_cache(samples: Sequence[dict], label: int, pair: bool):
    """Prepare score-sorted detections and overlap matrices once per class."""
    gt_by_sample = {}
    detections = []
    overlaps = {}
    for sample_id, sample in enumerate(samples):
        gt_labels = sample['gt_labels']
        gt_indices = torch.nonzero(gt_labels == label, as_tuple=False).flatten()
        gt_by_sample[sample_id] = gt_indices.tolist()
        pred_indices = torch.nonzero(sample['pred_labels'] == label,
                                     as_tuple=False).flatten()
        if not len(pred_indices) or not len(gt_indices):
            overlap = torch.empty((len(pred_indices), len(gt_indices)))
        elif pair:
            overlap = torch.full((len(pred_indices), len(gt_indices)), -1.0)
            for mode in ((True, True), (True, False), (False, True)):
                pred_mode = ((sample['pred_valid_prev'][pred_indices] == mode[0]) &
                             (sample['pred_valid_curr'][pred_indices] == mode[1]))
                gt_mode = ((sample['gt_valid_prev'][gt_indices] == mode[0]) &
                           (sample['gt_valid_curr'][gt_indices] == mode[1]))
                pred_rows = torch.nonzero(pred_mode, as_tuple=False).flatten()
                gt_cols = torch.nonzero(gt_mode, as_tuple=False).flatten()
                if not len(pred_rows) or not len(gt_cols):
                    continue
                pred_sel = pred_indices[pred_rows]
                gt_sel = gt_indices[gt_cols]
                ious = []
                if mode[0]:
                    ious.append(rbbox_overlaps(sample['pred_prev'][pred_sel],
                                                sample['gt_prev'][gt_sel]))
                if mode[1]:
                    ious.append(rbbox_overlaps(sample['pred_curr'][pred_sel],
                                                sample['gt_curr'][gt_sel]))
                overlap[pred_rows[:, None], gt_cols[None, :]] = torch.stack(ious).min(0).values
        else:
            overlap = rbbox_overlaps(sample['pred_boxes'][pred_indices],
                                     sample['gt_boxes'][gt_indices])
        overlaps[sample_id] = overlap.cpu().numpy()
        for pred_idx in pred_indices.tolist():
            detections.append((float(sample['pred_scores'][pred_idx]),
                               sample_id, pred_idx))
    num_gt = sum(len(indices) for indices in gt_by_sample.values())
    detections.sort(key=lambda item: item[0], reverse=True)
    pred_row = {
        sample_id: {pred_idx: row for row, pred_idx in enumerate(torch.nonzero(
            sample['pred_labels'] == label, as_tuple=False).flatten().tolist())}
        for sample_id, sample in enumerate(samples)}
    return num_gt, gt_by_sample, detections, overlaps, pred_row


def _class_aps(samples: Sequence[dict], label: int, pair: bool) -> List[float]:
    num_gt, gt_by_sample, detections, overlaps, pred_row = _class_cache(
        samples, label, pair)
    if num_gt == 0:
        return [float('nan')] * len(IOU_THRS)
    values = []
    for iou_thr in IOU_THRS:
        matched = {sample_id: set() for sample_id in gt_by_sample}
        tp = np.zeros(len(detections), dtype=np.float32)
        fp = np.zeros(len(detections), dtype=np.float32)
        for det_id, (_, sample_id, pred_idx) in enumerate(detections):
            gt_indices = gt_by_sample[sample_id]
            row = overlaps[sample_id][pred_row[sample_id][pred_idx]]
            best_iou, best_gt = -1.0, None
            for col, gt_idx in enumerate(gt_indices):
                if gt_idx not in matched[sample_id] and row[col] > best_iou:
                    best_iou, best_gt = row[col], gt_idx
            if best_gt is not None and best_iou >= iou_thr:
                matched[sample_id].add(best_gt)
                tp[det_id] = 1.0
            else:
                fp[det_id] = 1.0
        values.append(_ap_from_tp_fp(tp, fp, num_gt))
    return values


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
    pres_prev = _field(pred, 'presence_prev').detach().cpu().float().clamp(0, 1)
    pres_curr = _field(pred, 'presence_curr').detach().cpu().float().clamp(0, 1)
    pred_valid_prev = pres_prev >= pres_thr
    pred_valid_curr = pres_curr >= pres_thr
    # Score the visibility mode that the query explicitly predicts.
    pair_scores = cls_scores * torch.where(
        pred_valid_prev & pred_valid_curr, torch.sqrt(pres_prev * pres_curr),
        torch.where(pred_valid_prev, pres_prev * (1 - pres_curr),
                    (1 - pres_prev) * pres_curr))
    if max_dets is not None and pair_scores.numel() > max_dets:
        independent_scores = torch.maximum(cls_scores * pres_prev,
                                           cls_scores * pres_curr)
        rank_scores = torch.maximum(pair_scores, independent_scores)
        keep = torch.topk(rank_scores, k=max_dets).indices
        pred_labels = pred_labels[keep]
        pred_valid_prev = pred_valid_prev[keep]
        pred_valid_curr = pred_valid_curr[keep]
        pair_scores = pair_scores[keep]
        cls_scores = cls_scores[keep]
        pres_prev = pres_prev[keep]
        pres_curr = pres_curr[keep]
        pred_prev = to_rbox_tensor(_field(pred, 'bboxes_prev')).detach().cpu().float()[keep]
        pred_curr = to_rbox_tensor(_field(pred, 'bboxes_curr')).detach().cpu().float()[keep]
    else:
        pred_prev = to_rbox_tensor(_field(pred, 'bboxes_prev')).detach().cpu().float()
        pred_curr = to_rbox_tensor(_field(pred, 'bboxes_curr')).detach().cpu().float()
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
        pred_presence_prev=pres_prev,
        pred_presence_curr=pres_curr)


def pair_ap_metrics(samples: Sequence[dict]) -> Dict[str, float]:
    return _summary(samples, pair=True, prefix='pair')


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
                pred_scores=(sample['pred_cls_scores'] *
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
