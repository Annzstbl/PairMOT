# Copyright (c) AI4RS. All rights reserved.
"""Pair AP validation metrics for HSMOT pair RT-DETR."""

import csv
import math
import os
import os.path as osp
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from mmengine.structures import InstanceData
from mmrotate.registry import METRICS
from mmrotate.structures.bbox import qbox2rbox, rbbox_overlaps

from .pair_ap import (
    pair_and_independent_ap_metrics,
    pair_and_independent_ap_metrics_with_gt_filters,
    serialize_pair_sample,
)
from .pair_mot_tracker import (
    PairDetection,
    PairFrameRecord,
    PairMOTTracker,
    bootstrap_first_record_from_pair,
    write_pair_det_txt,
    write_trackeval_txt,
)

_METRIC_NOT_COMPUTED = float('nan')
_DIAGNOSTIC_METRIC_KEYS = (
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
)


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


def _maybe_field(data, key: str):
    if isinstance(data, dict):
        return data.get(key)
    return getattr(data, key, None)


def _scale_factor_list(meta: dict) -> List[float]:
    sf = meta.get('scale_factor', (1.0, 1.0))
    if isinstance(sf, torch.Tensor):
        sf = sf.detach().cpu().numpy()
    sf = np.asarray(sf, dtype=np.float32).reshape(-1)
    if sf.size == 1:
        sf = np.repeat(sf, 2)
    if sf.size >= 2:
        return [float(sf[0]), float(sf[1])]
    return [1.0, 1.0]


def _rboxes_to_original_image(rboxes: torch.Tensor, meta: dict) -> torch.Tensor:
    if rboxes.numel() == 0:
        return rboxes
    sx, sy = _scale_factor_list(meta)
    scale = rboxes.new_tensor([sx, sy, sx, sy, 1.0]).clamp(min=1e-6)
    return rboxes / scale


def _score_with_presence(cls_score: float,
                         presence_prev: Optional[float],
                         presence_curr: Optional[float],
                         mode: str) -> float:
    cls_score = float(cls_score)
    if mode == 'cls':
        return cls_score
    if presence_prev is None or presence_curr is None:
        return cls_score
    pres_min = min(float(presence_prev), float(presence_curr))
    if mode == 'cls_min_presence':
        return cls_score * pres_min
    if mode == 'auto':
        if max(float(presence_prev), float(presence_curr)) >= 0.5:
            return cls_score * pres_min
        return cls_score
    raise ValueError(f'Unsupported track score mode: {mode}')


def _track_record_from_sample(sample, pred, score_mode: str) -> Optional[dict]:
    if isinstance(sample, dict):
        meta = sample.get('metainfo', {}) or sample
    else:
        meta = getattr(sample, 'metainfo', {}) or {}
    required = ('seq_name', 'frame_id_prev', 'frame_id')
    if any(key not in meta for key in required):
        return None

    scores = _field(pred, 'scores').detach().cpu().float()
    labels = _field(pred, 'labels').detach().cpu().long()
    bboxes_prev = _rboxes_to_original_image(
        _field(pred, 'bboxes_prev').detach().cpu().float(), meta)
    bboxes_curr = _rboxes_to_original_image(
        _field(pred, 'bboxes_curr').detach().cpu().float(), meta)
    pres_prev = _maybe_field(pred, 'presence_prev')
    pres_curr = _maybe_field(pred, 'presence_curr')
    score_prev = _maybe_field(pred, 'scores_prev')
    score_curr = _maybe_field(pred, 'scores_curr')
    label_prev = _maybe_field(pred, 'labels_prev')
    label_curr = _maybe_field(pred, 'labels_curr')
    if pres_prev is not None:
        pres_prev = pres_prev.detach().cpu().float()
    if pres_curr is not None:
        pres_curr = pres_curr.detach().cpu().float()
    if score_prev is not None:
        score_prev = score_prev.detach().cpu().float()
    if score_curr is not None:
        score_curr = score_curr.detach().cpu().float()
    if label_prev is not None:
        label_prev = label_prev.detach().cpu().long()
    if label_curr is not None:
        label_curr = label_curr.detach().cpu().long()

    detections = []
    for idx in range(int(scores.numel())):
        pp = float(pres_prev[idx]) if pres_prev is not None else None
        pc = float(pres_curr[idx]) if pres_curr is not None else None
        cls_score = float(scores[idx])
        detections.append(dict(
            index=idx,
            prev_bbox=[float(x) for x in bboxes_prev[idx].tolist()],
            curr_bbox=[float(x) for x in bboxes_curr[idx].tolist()],
            score=_score_with_presence(cls_score, pp, pc, score_mode),
            cls_score=cls_score,
            label=int(labels[idx]),
            presence_prev=pp,
            presence_curr=pc,
            score_prev=(float(score_prev[idx]) if score_prev is not None
                        else None),
            score_curr=(float(score_curr[idx]) if score_curr is not None
                        else None),
            label_prev=(int(label_prev[idx]) if label_prev is not None
                        else None),
            label_curr=(int(label_curr[idx]) if label_curr is not None
                        else None),
        ))

    return dict(
        seq_name=str(meta['seq_name']),
        prev_frame_id=int(meta['frame_id_prev']),
        curr_frame_id=int(meta['frame_id']),
        frame_gap=int(meta.get('frame_gap', int(meta['frame_id']) -
                               int(meta['frame_id_prev']))),
        prev_img_path=str(meta.get('img_path_prev', '')),
        curr_img_path=str(meta.get('img_path', '')),
        img_shape=list(meta.get('img_shape', [])),
        ori_shape=list(meta.get('ori_shape', [])),
        scale_factor=_scale_factor_list(meta),
        detections=detections,
        is_first_pair=int(meta['frame_id']) == int(meta['frame_id_prev']),
    )


def _record_from_dict(record: dict) -> PairFrameRecord:
    data = dict(record)
    data['detections'] = [PairDetection(**det) for det in data['detections']]
    return PairFrameRecord(**data)


def _format_value(value: Union[float, int]) -> str:
    if isinstance(value, float) and math.isnan(value):
        return 'n/a'
    if isinstance(value, float):
        if value.is_integer():
            return f'{value:.0f}'
        return f'{value:.4f}'
    return str(value)


def _format_row(name: str, value: Union[float, int]) -> str:
    return f'| {name:<44} | {_format_value(value):>12} |'


def _not_computed_diagnostic_metrics() -> Dict[str, float]:
    return {key: _METRIC_NOT_COMPUTED for key in _DIAGNOSTIC_METRIC_KEYS}


def _filter_serialized_pair_sample(sample: dict, mode: str) -> dict:
    """Filter only GT fields in a serialized AP sample.

    Prediction tensors are intentionally reused.  This keeps the breakdown
    cheap enough for validation while making the formal ``pair_*`` metrics stay
    on all union GT.
    """
    if mode == 'all':
        return sample
    valid_prev = sample['gt_valid_prev'].bool()
    valid_curr = sample['gt_valid_curr'].bool()
    if mode == 'both':
        keep = valid_prev & valid_curr
    elif mode == 'new':
        keep = (~valid_prev) & valid_curr
    elif mode == 'disappear':
        keep = valid_prev & (~valid_curr)
    else:
        raise ValueError(f'Unsupported pair AP GT filter: {mode}')
    filtered = dict(sample)
    for key in ('gt_labels', 'gt_prev', 'gt_curr', 'gt_valid_prev',
                'gt_valid_curr'):
        filtered[key] = sample[key][keep]
    return filtered


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
    breakdown_keys = []
    for mode in ('both', 'new', 'disappear'):
        breakdown_keys.extend([
            f'{mode}_pair_AP50',
            f'{mode}_pair_mAP50_95',
            f'{mode}_independent_AP50',
            f'{mode}_independent_mAP50_95',
        ])
    if any(key in metrics for key in breakdown_keys):
        sections.append(('GT Filter AP Breakdown', breakdown_keys))
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


def _read_summary_row(path: str) -> dict:
    if not osp.isfile(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        row = next(csv.DictReader(f), None)
    return row or {}


def _add_trackeval_metrics(metrics: Dict[str, float], eval_dir: str) -> None:
    for src_name, prefix in (
            ('cls_comb_cls_av', 'track/cls'),
            ('cls_comb_det_av', 'track/det')):
        row = _read_summary_row(osp.join(eval_dir, f'{src_name}_summary.csv'))
        for metric_name in ('HOTA', 'MOTA', 'IDF1'):
            if metric_name in row:
                metrics[f'{prefix}_{metric_name.lower()}'] = float(
                    row[metric_name])

    for filename in sorted(os.listdir(eval_dir)) if osp.isdir(eval_dir) else []:
        if not filename.endswith('_summary.csv'):
            continue
        name = filename[:-len('_summary.csv')]
        if name in {
                'cls_comb_cls_av', 'cls_comb_det_av', 'all_cls', 'all_seq',
                'BIKE', 'HUMAN', 'VEHICLE'
        }:
            continue
        row = _read_summary_row(osp.join(eval_dir, filename))
        safe_name = name.replace('/', '_').replace(' ', '_')
        for metric_name in ('HOTA', 'MOTA', 'IDF1'):
            if metric_name in row:
                metrics[
                    f'track_class/{safe_name}_{metric_name.lower()}'] = float(
                        row[metric_name])


def _run_track_eval_from_records(records: Sequence[dict], *, out_dir: str,
                                 data_root: str, ann_subdir: str,
                                 img_subdir: str, trackeval_root: str,
                                 tracker_name: str, tracker_sub_folder: str,
                                 new_born_th: float, track_th: float,
                                 match_iou_th: float,
                                 new_birth_iou_th: float,
                                 max_age: int,
                                 init_same_iou_th: float,
                                 class_aware: bool) -> Dict[str, float]:
    pair_records = [_record_from_dict(record) for record in records]
    by_seq: Dict[str, List[PairFrameRecord]] = {}
    for record in pair_records:
        if record.frame_gap != 1:
            continue
        by_seq.setdefault(record.seq_name, []).append(record)

    tracker_root = osp.join(out_dir, 'trackers', tracker_name)
    pred_dir = osp.join(tracker_root, tracker_sub_folder)
    val_det_dir = osp.join(out_dir, 'val_det')
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(val_det_dir, exist_ok=True)

    for seq_name, seq_records in sorted(by_seq.items()):
        seq_records = sorted(
            seq_records, key=lambda item: (item.prev_frame_id,
                                           item.curr_frame_id))
        if not seq_records:
            continue
        write_pair_det_txt(osp.join(val_det_dir, f'{seq_name}.txt'), seq_records)
        tracker = PairMOTTracker(
            new_born_th=new_born_th,
            track_th=track_th,
            match_iou_th=match_iou_th,
            new_birth_iou_th=new_birth_iou_th,
            max_age=max_age,
            init_same_iou_th=init_same_iou_th,
            class_aware=class_aware)
        tracker.init_first_frame(bootstrap_first_record_from_pair(seq_records[0]))
        for record in seq_records:
            tracker.update_pair(record)
        write_trackeval_txt(
            osp.join(pred_dir, f'{seq_name}.txt'), tracker.all_history())

    cmd = [
        sys.executable,
        osp.join(osp.abspath(trackeval_root), 'scripts/run_hsmot_8ch.py'),
        '--USE_PARALLEL', 'False',
        '--METRICS', 'HOTA', 'CLEAR', 'Identity',
        '--TRACKERS_TO_EVAL', tracker_name,
        '--TRACKER_SUB_FOLDER', tracker_sub_folder,
        '--GT_FOLDER', osp.abspath(osp.join(data_root, ann_subdir)),
        '--IMG_FOLDER', osp.abspath(osp.join(data_root, img_subdir)),
        '--TRACKERS_FOLDER', osp.abspath(osp.join(out_dir, 'trackers')),
        '--OUTPUT_FOLDER', osp.abspath(osp.join(out_dir, 'trackers')),
    ]
    env = os.environ.copy()
    ai4rs_root = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
    pairmmot_root = osp.abspath(osp.join(ai4rs_root, '..'))
    extra_pythonpath = [
        ai4rs_root,
        pairmmot_root,
        osp.join(pairmmot_root, 'hsmot'),
    ]
    env['PYTHONPATH'] = os.pathsep.join(
        extra_pythonpath + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    subprocess.run(
        cmd, cwd=osp.abspath(trackeval_root), check=True, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    metrics: Dict[str, float] = {}
    _add_trackeval_metrics(metrics, osp.join(tracker_root, 'eval'))
    return metrics


@METRICS.register_module()
class HSMOTPairOverfitMetric(BaseMetric):
    """Backward-compatible registry name for pair AP validation.

    Set ``diagnostic_mode=True`` to run the expensive per-sample matching
    counters used by the Matching Diagnostics table.  Formal training keeps
    this disabled and only computes ranking AP.
    """

    default_prefix = 'pair'

    def __init__(self,
                 score_thr: float = 0.35,
                 iou_thr: float = 0.5,
                 pres_thr: float = 0.5,
                 max_dets: int | None = 100,
                 report_gaps: Sequence[int] = (),
                 both_visible_gt_only: bool = False,
                 diagnostic_mode: bool = False,
                 track_eval: bool = False,
                 track_eval_out_dir: Optional[str] = None,
                 trackeval_root: str = '/data/users/litianhao01/PairMmot/TrackEval',
                 track_data_root: str = '/data/users/litianhao01/PairMmot/data/hsmot/test',
                 track_ann_subdir: str = 'mot',
                 track_img_subdir: str = 'npy2jpg',
                 track_tracker_sub_folder: str = 'preds',
                 track_score_mode: str = 'cls',
                 track_new_born_th: float = 0.6,
                 track_track_th: float = 0.2,
                 track_match_iou_th: float = 0.25,
                 track_new_birth_iou_th: float = 0.5,
                 track_max_age: int = 30,
                 track_init_same_iou_th: float = 0.3,
                 track_class_aware: bool = False,
                 collect_device: str = 'cpu',
                 prefix: str = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.pres_thr = pres_thr
        self.max_dets = max_dets
        self.report_gaps = tuple(sorted(set(int(gap) for gap in report_gaps)))
        self.both_visible_gt_only = bool(both_visible_gt_only)
        self.diagnostic_mode = bool(diagnostic_mode)
        self.track_eval = bool(track_eval)
        self.track_eval_out_dir = track_eval_out_dir
        self.trackeval_root = trackeval_root
        self.track_data_root = track_data_root
        self.track_ann_subdir = track_ann_subdir
        self.track_img_subdir = track_img_subdir
        self.track_tracker_sub_folder = track_tracker_sub_folder
        self.track_score_mode = track_score_mode
        self.track_new_born_th = float(track_new_born_th)
        self.track_track_th = float(track_track_th)
        self.track_match_iou_th = float(track_match_iou_th)
        self.track_new_birth_iou_th = float(track_new_birth_iou_th)
        self.track_max_age = int(track_max_age)
        self.track_init_same_iou_th = float(track_init_same_iou_th)
        self.track_class_aware = bool(track_class_aware)
        self._track_eval_count = 0

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
            stats = dict(
                ap_sample=serialize_pair_sample(
                    gt, pred, pres_thr=self.pres_thr, max_dets=self.max_dets),
                frame_gap=int(getattr(sample, 'metainfo', {}).get(
                    'frame_gap', 0)),
            )
            if self.track_eval:
                track_record = _track_record_from_sample(
                    sample, pred, self.track_score_mode)
                if track_record is not None:
                    stats['track_record'] = track_record
            if self.diagnostic_mode:
                diagnostic_gt = self._filter_gt(gt)
                stats.update(_eval_pair_sample(
                    diagnostic_gt,
                    pred,
                    score_thr=self.score_thr,
                    iou_thr=self.iou_thr,
                    pres_thr=self.pres_thr,
                ))
            self.results.append(stats)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        if not results:
            metrics = _not_computed_diagnostic_metrics()
            if self.diagnostic_mode:
                metrics.update(
                    gt_pairs=0.0,
                    match_ratio=0.0,
                    mean_iou_prev=0.0,
                    mean_iou_curr=0.0,
                    presence_acc=0.0,
                )
            return metrics

        ap_samples = [r['ap_sample'] for r in results]
        if self.diagnostic_mode:
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
        else:
            metrics = _not_computed_diagnostic_metrics()
        ap_metrics = pair_and_independent_ap_metrics_with_gt_filters(
            ap_samples, gt_filters=('both', 'new', 'disappear'))
        metrics.update(ap_metrics)
        pair_metrics = {
            key: value
            for key, value in ap_metrics.items()
            if key.startswith('pair_')
        }
        independent_metrics = {
            key: value
            for key, value in ap_metrics.items()
            if key.startswith('independent_') or key == 'association_gap_AP50'
        }
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
            gap_metrics = pair_and_independent_ap_metrics(gap_samples)
            metrics.update({
                f'{prefix}{name}': value
                for name, value in gap_metrics.items()
            })
        print_log(_format_pair_metric_table(metrics), logger='current')
        if self.track_eval:
            track_records = [
                r['track_record'] for r in results if 'track_record' in r]
            metrics['track/num_records'] = float(len(track_records))
            metrics['track/num_sequences'] = float(
                len({record['seq_name'] for record in track_records}))
            if track_records and self.track_eval_out_dir:
                self._track_eval_count += 1
                eval_out_dir = osp.join(
                    self.track_eval_out_dir,
                    f'val_track_{self._track_eval_count:04d}')
                tracker_name = f'val_pairmot_{self._track_eval_count:04d}'
                try:
                    track_metrics = _run_track_eval_from_records(
                        track_records,
                        out_dir=eval_out_dir,
                        data_root=self.track_data_root,
                        ann_subdir=self.track_ann_subdir,
                        img_subdir=self.track_img_subdir,
                        trackeval_root=self.trackeval_root,
                        tracker_name=tracker_name,
                        tracker_sub_folder=self.track_tracker_sub_folder,
                        new_born_th=self.track_new_born_th,
                        track_th=self.track_track_th,
                        match_iou_th=self.track_match_iou_th,
                        new_birth_iou_th=self.track_new_birth_iou_th,
                        max_age=self.track_max_age,
                        init_same_iou_th=self.track_init_same_iou_th,
                        class_aware=self.track_class_aware)
                    metrics.update(track_metrics)
                    print_log(
                        f'Pair MOT validation summary written to {eval_out_dir}',
                        logger='current')
                except Exception as exc:
                    print_log(
                        f'Pair MOT validation failed: {exc}',
                        logger='current')
            else:
                print_log(
                    'Pair MOT validation skipped: '
                    f'track_records={len(track_records)}, '
                    f'track_eval_out_dir={self.track_eval_out_dir}',
                    logger='current')
        return metrics


@METRICS.register_module()
class HSMOTPairAPMetric(HSMOTPairOverfitMetric):
    """Production name for the pair AP evaluator.

    The historical ``HSMOTPairOverfitMetric`` name remains registered so old
    acceptance configs continue to run unchanged.
    """

    pass
