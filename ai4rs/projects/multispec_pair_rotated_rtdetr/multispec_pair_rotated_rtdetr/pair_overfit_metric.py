# Copyright (c) AI4RS. All rights reserved.
"""Pair overfit validation metrics for HSMOT pair RT-DETR."""

from typing import Dict, List, Sequence

import torch
from mmengine.evaluator import BaseMetric
from mmengine.structures import InstanceData
from mmrotate.registry import METRICS
from mmrotate.structures.bbox import qbox2rbox, rbbox_overlaps


def _to_rbox_tensor(boxes) -> torch.Tensor:
    if hasattr(boxes, 'tensor'):
        tensor = boxes.tensor
    else:
        tensor = boxes
    if tensor.size(-1) == 8:
        return qbox2rbox(tensor)
    return tensor


def _rbox_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return float(rbbox_overlaps(
        a.unsqueeze(0), b.unsqueeze(0), is_aligned=True)[0].item())


def _eval_pair_sample(
    gt: InstanceData,
    pred: InstanceData,
    *,
    score_thr: float,
    iou_thr: float,
    pres_thr: float,
) -> Dict[str, float]:
    """Evaluate one pair sample; return per-sample counters."""
    num_gt = len(gt.labels)
    stats = dict(
        gt_pairs=float(num_gt),
        matched_queries=0.0,
        duplicate_match=0.0,
        iou_prev_sum=0.0,
        iou_curr_sum=0.0,
        iou_prev_count=0.0,
        iou_curr_count=0.0,
        presence_ok=0.0,
        presence_total=0.0,
        match_fail=0.0,
        iou_prev_fail=0.0,
        iou_curr_fail=0.0,
        presence_fail=0.0,
    )
    if num_gt == 0:
        return stats

    gt_labels = gt.labels.cpu()
    gt_prev = _to_rbox_tensor(gt.bboxes_prev).cpu()
    gt_curr = _to_rbox_tensor(gt.bboxes_curr).cpu()
    valid_prev = gt.valid_prev.cpu().bool()
    valid_curr = gt.valid_curr.cpu().bool()

    pred_scores = pred.scores.cpu()
    pred_labels = pred.labels.cpu()
    pred_prev = pred.bboxes_prev.cpu()
    pred_curr = pred.bboxes_curr.cpu()
    pred_pres_p = pred.presence_prev.cpu()
    pred_pres_c = pred.presence_curr.cpu()

    used_queries = set()
    for gi in range(num_gt):
        label = int(gt_labels[gi].item())
        cls_mask = pred_labels == label
        cand_scores = pred_scores.clone()
        cand_scores[~cls_mask] = -1.0
        cand_scores[list(used_queries)] = -1.0
        best_q = int(cand_scores.argmax().item())
        best_score = float(cand_scores[best_q].item())

        if best_score < score_thr:
            stats['match_fail'] += 1.0
            continue

        if best_q in used_queries:
            stats['duplicate_match'] += 1.0
        used_queries.add(best_q)
        stats['matched_queries'] += 1.0

        if valid_prev[gi]:
            iou_p = _rbox_iou(pred_prev[best_q], gt_prev[gi])
            stats['iou_prev_sum'] += iou_p
            stats['iou_prev_count'] += 1.0
            if iou_p < iou_thr:
                stats['iou_prev_fail'] += 1.0
        elif pred_pres_p[best_q].item() > pres_thr:
            stats['presence_fail'] += 1.0
        stats['presence_total'] += 1.0
        stats['presence_ok'] += float(
            (pred_pres_p[best_q].item() > pres_thr) == bool(
                valid_prev[gi].item()))

        if valid_curr[gi]:
            iou_c = _rbox_iou(pred_curr[best_q], gt_curr[gi])
            stats['iou_curr_sum'] += iou_c
            stats['iou_curr_count'] += 1.0
            if iou_c < iou_thr:
                stats['iou_curr_fail'] += 1.0
        elif pred_pres_c[best_q].item() > pres_thr:
            stats['presence_fail'] += 1.0
        stats['presence_total'] += 1.0
        stats['presence_ok'] += float(
            (pred_pres_c[best_q].item() > pres_thr) == bool(
                valid_curr[gi].item()))

    return stats


@METRICS.register_module()
class HSMOTPairOverfitMetric(BaseMetric):
    """Pair matching / IoU / presence metrics for overfit validation."""

    default_prefix = 'pair'

    def __init__(self,
                 score_thr: float = 0.35,
                 iou_thr: float = 0.5,
                 pres_thr: float = 0.5,
                 collect_device: str = 'cpu',
                 prefix: str = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.pres_thr = pres_thr

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            if not hasattr(sample, 'pair_gt_instances'):
                continue
            if not hasattr(sample, 'pred_pair_instances'):
                continue
            self.results.append(
                _eval_pair_sample(
                    sample.pair_gt_instances,
                    sample.pred_pair_instances,
                    score_thr=self.score_thr,
                    iou_thr=self.iou_thr,
                    pres_thr=self.pres_thr,
                ))

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        if not results:
            return dict(
                gt_pairs=0.0,
                match_ratio=0.0,
                mean_iou_prev=0.0,
                mean_iou_curr=0.0,
                presence_acc=0.0,
            )

        total_gt = sum(r['gt_pairs'] for r in results)
        matched = sum(r['matched_queries'] for r in results)
        iou_prev_sum = sum(r['iou_prev_sum'] for r in results)
        iou_curr_sum = sum(r['iou_curr_sum'] for r in results)
        iou_prev_count = sum(r['iou_prev_count'] for r in results)
        iou_curr_count = sum(r['iou_curr_count'] for r in results)
        presence_ok = sum(r['presence_ok'] for r in results)
        presence_total = sum(r['presence_total'] for r in results)

        metrics = dict(
            gt_pairs=total_gt,
            matched_queries=matched,
            match_ratio=matched / max(total_gt, 1.0),
            duplicate_match=sum(r['duplicate_match'] for r in results),
            match_fail=sum(r['match_fail'] for r in results),
            iou_prev_fail=sum(r['iou_prev_fail'] for r in results),
            iou_curr_fail=sum(r['iou_curr_fail'] for r in results),
            presence_fail=sum(r['presence_fail'] for r in results),
        )
        if iou_prev_count > 0:
            metrics['mean_iou_prev'] = iou_prev_sum / iou_prev_count
        if iou_curr_count > 0:
            metrics['mean_iou_curr'] = iou_curr_sum / iou_curr_count
        if presence_total > 0:
            metrics['presence_acc'] = presence_ok / presence_total
        return metrics
