# Copyright (c) AI4RS. All rights reserved.
"""Pair overfit validation metrics for HSMOT pair RT-DETR."""

from typing import Dict, List, Sequence

import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from mmengine.structures import InstanceData
from mmrotate.registry import METRICS
from mmrotate.structures.bbox import qbox2rbox, rbbox_overlaps

from .overfit_ap import independent_ap_metrics, pair_ap_metrics, serialize_pair_sample


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


def _field(data, key: str):
    if isinstance(data, dict):
        return data[key]
    return getattr(data, key)


def _format_value(value: float) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return f'{value:.0f}'
        return f'{value:.4f}'
    return str(value)


def _format_row(name: str, value: float) -> str:
    return f'| {name:<44} | {_format_value(value):>12} |'


def _format_pair_metric_table(metrics: Dict[str, float]) -> str:
    """Build a compact validation summary for human-readable logs."""
    sections = [
        ('Detection AP', [
            'independent_AP50',
            'independent_AP75',
            'independent_mAP50_95',
            'independent_prev_AP50',
            'independent_curr_AP50',
        ]),
        ('Pair AP', [
            'pair_AP50',
            'pair_AP75',
            'pair_mAP50_95',
            'association_gap_AP50',
        ]),
        ('Matching Diagnostics', [
            'gt_pairs',
            'matched_queries',
            'match_ratio',
            'duplicate_match',
            'match_fail',
            'iou_prev_fail',
            'iou_curr_fail',
            'presence_fail',
            'mean_iou_prev',
            'mean_iou_curr',
            'presence_acc',
        ]),
    ]
    gap_prefixes = sorted({
        key.split('_', 1)[0]
        for key in metrics
        if key.startswith('gap') and '_' in key
    })
    for gap_prefix in gap_prefixes:
        sections.append((f'{gap_prefix} AP', [
            f'{gap_prefix}_independent_AP50',
            f'{gap_prefix}_pair_AP50',
            f'{gap_prefix}_association_gap_AP50',
            f'{gap_prefix}_independent_mAP50_95',
            f'{gap_prefix}_pair_mAP50_95',
        ]))
    class_keys = sorted(
        [key for key in metrics if '_class' in key and key.endswith('_AP50')])
    if class_keys:
        sections.append(('Class AP50', class_keys))

    lines = [
        '',
        'Pair validation summary:',
        '+----------------------------------------------+--------------+',
        '| metric                                       |        value |',
        '+----------------------------------------------+--------------+',
    ]
    for section_name, keys in sections:
        present_keys = [key for key in keys if key in metrics]
        if not present_keys:
            continue
        lines.append(f'| [{section_name:<42}] |              |')
        for key in present_keys:
            lines.append(_format_row(key, metrics[key]))
        lines.append('+----------------------------------------------+--------------+')
    return '\n'.join(lines)


def _eval_pair_sample(
    gt: InstanceData,
    pred: InstanceData,
    *,
    score_thr: float,
    iou_thr: float,
    pres_thr: float,
) -> Dict[str, float]:
    """Evaluate one pair sample; return per-sample counters."""
    gt_labels_data = _field(gt, 'labels')
    num_gt = len(gt_labels_data)
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

    gt_labels = gt_labels_data.cpu()
    gt_prev = _to_rbox_tensor(_field(gt, 'bboxes_prev')).cpu()
    gt_curr = _to_rbox_tensor(_field(gt, 'bboxes_curr')).cpu()
    valid_prev = _field(gt, 'valid_prev').cpu().bool()
    valid_curr = _field(gt, 'valid_curr').cpu().bool()

    pred_scores = _field(pred, 'scores').cpu()
    pred_labels = _field(pred, 'labels').cpu()
    pred_prev = _field(pred, 'bboxes_prev').cpu()
    pred_curr = _field(pred, 'bboxes_curr').cpu()
    has_presence = hasattr(pred, 'presence_prev') or (
        isinstance(pred, dict) and 'presence_prev' in pred)
    if has_presence:
        pred_pres_p = _field(pred, 'presence_prev').cpu()
        pred_pres_c = _field(pred, 'presence_curr').cpu()
    else:
        pred_pres_p = torch.ones_like(pred_scores)
        pred_pres_c = torch.ones_like(pred_scores)

    candidates = []
    has_score_candidate = [False] * num_gt
    for gi in range(num_gt):
        label = int(gt_labels[gi].item())
        for qi in range(len(pred_scores)):
            if int(pred_labels[qi].item()) != label:
                continue
            if float(pred_scores[qi].item()) < score_thr:
                continue
            has_score_candidate[gi] = True
            ious = []
            if valid_prev[gi]:
                ious.append(_rbox_iou(pred_prev[qi], gt_prev[gi]))
            if valid_curr[gi]:
                ious.append(_rbox_iou(pred_curr[qi], gt_curr[gi]))
            mean_iou = sum(ious) / len(ious) if ious else 0.0
            candidates.append((mean_iou, gi, qi))

    gt_to_query = {}
    used_gt = set()
    used_queries = set()
    for mean_iou, gi, qi in sorted(candidates, reverse=True):
        if gi in used_gt or qi in used_queries:
            continue
        used_gt.add(gi)
        used_queries.add(qi)
        gt_to_query[gi] = qi

    for gi in range(num_gt):
        if gi not in gt_to_query:
            stats['match_fail'] += 1.0
            if has_score_candidate[gi]:
                stats['duplicate_match'] += 1.0
            continue

        best_q = gt_to_query[gi]
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
    """Pair matching / IoU / presence metrics and ranking AP validation."""

    default_prefix = 'pair'

    def __init__(self,
                 score_thr: float = 0.35,
                 iou_thr: float = 0.5,
                 pres_thr: float = 0.5,
                 max_dets: int | None = 100,
                 report_gaps: Sequence[int] = (),
                 both_visible_gt_only: bool = False,
                 collect_device: str = 'cpu',
                 prefix: str = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.pres_thr = pres_thr
        self.max_dets = max_dets
        self.report_gaps = tuple(sorted(set(int(gap) for gap in report_gaps)))
        self.both_visible_gt_only = bool(both_visible_gt_only)

    def _filter_gt(self, gt):
        if not self.both_visible_gt_only:
            return gt
        valid = _field(gt, 'valid_prev').bool() & _field(gt, 'valid_curr').bool()
        if valid.all():
            return gt
        if isinstance(gt, dict):
            filtered = {}
            for key, value in gt.items():
                try:
                    should_filter = (
                        hasattr(value, '__getitem__') and len(value) == len(valid))
                except TypeError:
                    should_filter = False
                filtered[key] = value[valid] if should_filter else value
        else:
            filtered = InstanceData()
            for key in gt.keys():
                value = getattr(gt, key)
                try:
                    should_filter = (
                        hasattr(value, '__getitem__') and len(value) == len(valid))
                except TypeError:
                    should_filter = False
                if should_filter:
                    setattr(filtered, key, value[valid])
                else:
                    setattr(filtered, key, value)
        return filtered

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            if isinstance(sample, dict):
                gt = sample.get('pair_gt_instances')
                pred = sample.get('pred_pair_instances')
            else:
                gt = getattr(sample, 'pair_gt_instances', None)
                pred = getattr(sample, 'pred_pair_instances', None)
            if gt is None:
                continue
            if pred is None:
                continue
            gt = self._filter_gt(gt)
            # Keep the legacy counters as diagnostics, but make AP the metric
            # reported by validation and used by the acceptance scripts.
            stats = _eval_pair_sample(
                gt,
                pred,
                score_thr=self.score_thr,
                iou_thr=self.iou_thr,
                pres_thr=self.pres_thr,
            )
            stats['ap_sample'] = serialize_pair_sample(
                gt, pred, pres_thr=self.pres_thr, max_dets=self.max_dets)
            stats['frame_gap'] = int(getattr(sample, 'metainfo', {}).get(
                'frame_gap', 0))
            self.results.append(stats)

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
        ap_samples = [r['ap_sample'] for r in results]
        pair_metrics = pair_ap_metrics(ap_samples)
        metrics.update(pair_metrics)
        independent_metrics = independent_ap_metrics(
            ap_samples, pair_ap50=pair_metrics['pair_AP50'])
        metrics.update(independent_metrics)
        for gap in self.report_gaps:
            gap_samples = [r['ap_sample'] for r in results
                           if r.get('frame_gap') == gap]
            if not gap_samples:
                continue
            prefix = f'gap{gap}_'
            if len(gap_samples) == len(ap_samples):
                metrics.update({
                    f'{prefix}{name}': value
                    for name, value in pair_metrics.items()
                })
                metrics.update({
                    f'{prefix}{name}': value
                    for name, value in independent_metrics.items()
                })
                continue
            gap_pair_metrics = pair_ap_metrics(gap_samples)
            metrics.update({
                f'{prefix}{name}': value
                for name, value in gap_pair_metrics.items()
            })
            metrics.update({
                f'{prefix}{name}': value
                for name, value in independent_ap_metrics(
                    gap_samples,
                    pair_ap50=gap_pair_metrics['pair_AP50']).items()
            })
        print_log(_format_pair_metric_table(metrics), logger='current')
        return metrics


@METRICS.register_module()
class HSMOTPairAPMetric(HSMOTPairOverfitMetric):
    """Production name for the pair AP evaluator.

    The historical ``HSMOTPairOverfitMetric`` name remains registered so old
    acceptance configs continue to run unchanged.
    """

    pass
