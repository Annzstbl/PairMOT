# Copyright (c) AI4RS. All rights reserved.
"""Validation visualization hook for HSMOT pair RT-DETR."""

import os.path as osp
from collections import defaultdict
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from mmengine.dist import get_dist_info
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.structures import InstanceData
from mmengine.utils import mkdir_or_exist
from mmrotate.registry import HOOKS
from mmrotate.structures.bbox import rbox2qbox

from mmrotate.datasets.transforms.visualize_hsmot_pair import (
    _to_numpy_image,
    visualize_hsmot_pair,
)

from .pair_overfit_metric import _eval_pair_sample, _rbox_iou, _to_rbox_tensor


def _crop_frame_to_img_shape(frame: torch.Tensor,
                             img_shape: tuple) -> torch.Tensor:
    """Drop bottom-right pad so the canvas matches ``img_shape``."""
    h, w = int(img_shape[0]), int(img_shape[1])
    return frame[..., :h, :w]


def _scale_factor_tensor(img_meta: dict, device, dtype) -> torch.Tensor:
    """Build per-rbox scale vector from pipeline ``scale_factor``."""
    sf = img_meta.get('scale_factor', (1.0, 1.0))
    if isinstance(sf, torch.Tensor):
        sf = sf.detach().cpu().numpy()
    sf = np.asarray(sf, dtype=np.float32).reshape(-1)
    if sf.size == 1:
        sf = np.repeat(sf, 2)
    if sf.size == 2:
        sf = np.array([sf[0], sf[1], sf[0], sf[1], 1.0], dtype=np.float32)
    elif sf.size == 4:
        sf = np.append(sf, 1.0)
    return torch.tensor(sf, device=device, dtype=dtype)


def _bboxes_to_input_space(bboxes: torch.Tensor,
                           img_meta: dict) -> torch.Tensor:
    """Map boxes from original-image space back to network ``img_shape``."""
    if bboxes.numel() == 0:
        return bboxes
    img_h, img_w = img_meta['img_shape']
    boxes = bboxes.clone()
    max_xy = float(boxes[:, 0:4:2].max().item()) if boxes.size(0) > 0 else 0.0
    max_yx = float(boxes[:, 1:4:2].max().item()) if boxes.size(0) > 0 else 0.0
    if max_xy <= img_w * 1.05 and max_yx <= img_h * 1.05:
        return boxes
    scale = _scale_factor_tensor(img_meta, boxes.device, boxes.dtype)
    return boxes * scale


def _pair_frames_for_vis(pair_input: torch.Tensor,
                         img_shape: tuple) -> tuple:
    """Crop prev/curr padded tensors to resized ``img_shape`` for drawing."""
    prev = _crop_frame_to_img_shape(pair_input[0], img_shape)
    curr = _crop_frame_to_img_shape(pair_input[1], img_shape)
    return prev, curr


def _polygons_from_rboxes(bboxes: torch.Tensor) -> np.ndarray:
    if bboxes.numel() == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    qbox = rbox2qbox(bboxes)
    return qbox.reshape(-1, 4, 2).cpu().numpy()


def _draw_pred_boxes(canvas: np.ndarray,
                     bboxes: torch.Tensor,
                     labels: Sequence[str],
                     color: tuple) -> None:
    polys = _polygons_from_rboxes(bboxes)
    for poly, label in zip(polys, labels):
        pts = poly.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        cx = int(np.clip(poly[:, 0].mean(), 0, canvas.shape[1] - 1))
        cy = int(np.clip(poly[:, 1].mean(), 0, canvas.shape[0] - 1))
        cv2.putText(canvas, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)


def _match_pred_indices(
    gt: InstanceData,
    pred: InstanceData,
    score_thr: float,
) -> dict:
    """Legacy score-first matcher retained for single-frame visualization."""
    matches = {}
    if len(gt.labels) == 0:
        return matches
    gt_labels = gt.labels.cpu()
    pred_scores = pred.scores.cpu()
    pred_labels = pred.labels.cpu()
    used_queries = set()
    for gi in range(len(gt_labels)):
        label = int(gt_labels[gi].item())
        candidates = pred_scores.clone()
        candidates[pred_labels != label] = -1.0
        candidates[list(used_queries)] = -1.0
        best_q = int(candidates.argmax().item())
        if float(candidates[best_q].item()) < score_thr:
            matches[gi] = -1
            continue
        used_queries.add(best_q)
        matches[gi] = best_q
    return matches


def _match_pred_indices_by_iou(
    gt: InstanceData,
    pred: InstanceData,
    iou_thr: float,
) -> dict:
    """Assign geometrically valid same-class queries for GT diagnostics."""
    gt_labels = gt.labels.cpu()
    gt_prev = _to_rbox_tensor(gt.bboxes_prev).cpu()
    gt_curr = _to_rbox_tensor(gt.bboxes_curr).cpu()
    valid_prev = gt.valid_prev.cpu().bool()
    valid_curr = gt.valid_curr.cpu().bool()
    pred_prev = _to_rbox_tensor(pred.bboxes_prev).cpu()
    pred_curr = _to_rbox_tensor(pred.bboxes_curr).cpu()
    pred_labels = pred.labels.cpu()
    candidates = []
    for gi, label in enumerate(gt_labels.tolist()):
        for qi, pred_label in enumerate(pred_labels.tolist()):
            if pred_label != label:
                continue
            ious = []
            if valid_prev[gi]:
                ious.append(_rbox_iou(pred_prev[qi], gt_prev[gi]))
            if valid_curr[gi]:
                ious.append(_rbox_iou(pred_curr[qi], gt_curr[gi]))
            pair_iou = min(ious) if ious else 0.0
            if pair_iou >= iou_thr:
                candidates.append((pair_iou, gi, qi))

    matches = {}
    used_queries = set()
    used_gt = set()
    for pair_iou, gi, qi in sorted(candidates, reverse=True):
        if gi in used_gt or qi in used_queries:
            continue
        matches[gi] = qi
        used_gt.add(gi)
        used_queries.add(qi)
    return matches


def _draw_prediction_view(
    left: np.ndarray,
    right: np.ndarray,
    pred: InstanceData,
    *,
    score_thr: float,
    pres_thr: float,
    img_meta: Optional[dict],
    color: tuple,
) -> None:
    """Draw all deployment candidates above a score threshold."""
    pred_prev = _to_rbox_tensor(pred.bboxes_prev).cpu()
    pred_curr = _to_rbox_tensor(pred.bboxes_curr).cpu()
    if img_meta is not None:
        pred_prev = _bboxes_to_input_space(pred_prev, img_meta).cpu()
        pred_curr = _bboxes_to_input_space(pred_curr, img_meta).cpu()
    scores = pred.scores.cpu()
    pres_prev = pred.presence_prev.cpu()
    pres_curr = pred.presence_curr.cpu()
    keep = scores >= score_thr
    labels = [
        f'p{qi} s={scores[qi]:.2f} pr={pres_prev[qi]:.2f}'
        for qi in torch.nonzero(keep & (pres_prev >= pres_thr),
                                 as_tuple=False).flatten().tolist()
    ]
    prev_keep = keep & (pres_prev >= pres_thr)
    curr_keep = keep & (pres_curr >= pres_thr)
    _draw_pred_boxes(left, pred_prev[prev_keep], labels, color)
    labels = [
        f'p{qi} s={scores[qi]:.2f} pr={pres_curr[qi]:.2f}'
        for qi in torch.nonzero(curr_keep, as_tuple=False).flatten().tolist()
    ]
    _draw_pred_boxes(right, pred_curr[curr_keep], labels, color)


def visualize_hsmot_pair_pred_gt(
    img_prev,
    img_curr,
    gt: InstanceData,
    pred: InstanceData,
    *,
    score_thr: float = 0.35,
    iou_thr: float = 0.5,
    pres_thr: float = 0.5,
    save_path: Optional[str] = None,
    meta_line: Optional[str] = None,
    img_meta: Optional[dict] = None,
    view: str = 'deploy',
    low_score_thr: float = 0.10,
) -> np.ndarray:
    """Draw GT plus a deployment, low-score, or IoU diagnostic view."""
    stats = _eval_pair_sample(
        gt,
        pred,
        score_thr=score_thr,
        iou_thr=iou_thr,
        pres_thr=pres_thr,
    )
    match_ratio = stats['matched_queries'] / max(stats['gt_pairs'], 1.0)
    summary = (
        f'match={stats["matched_queries"]:.0f}/{stats["gt_pairs"]:.0f} '
        f'({match_ratio:.2f})')
    if stats['iou_prev_count'] > 0:
        mean_p = stats['iou_prev_sum'] / stats['iou_prev_count']
        mean_c = stats['iou_curr_sum'] / max(stats['iou_curr_count'], 1.0)
        summary += f' iou_prev={mean_p:.2f} iou_curr={mean_c:.2f}'
    if stats['presence_total'] > 0:
        pres_acc = stats['presence_ok'] / stats['presence_total']
        summary += f' pres_acc={pres_acc:.2f}'

    vis = visualize_hsmot_pair(
        img_prev,
        img_curr,
        gt.bboxes_prev,
        gt.bboxes_curr,
        gt.track_ids.tolist(),
        gt.valid_prev.tolist(),
        gt.valid_curr.tolist(),
        save_path=None,
        meta_line=meta_line,
        check_summary=summary,
    )

    h = vis.shape[0]
    w_half = vis.shape[1] // 2
    left = vis[:, :w_half].copy()
    right = vis[:, w_half:].copy()

    if view == 'deploy':
        _draw_prediction_view(
            left, right, pred, score_thr=score_thr, pres_thr=pres_thr,
            img_meta=img_meta, color=(0, 165, 255))
    elif view == 'low_score':
        _draw_prediction_view(
            left, right, pred, score_thr=low_score_thr, pres_thr=pres_thr,
            img_meta=img_meta, color=(0, 255, 255))
    elif view == 'iou_diag':
        matches = _match_pred_indices_by_iou(gt, pred, iou_thr=iou_thr)
        pred_prev = _to_rbox_tensor(pred.bboxes_prev).cpu()
        pred_curr = _to_rbox_tensor(pred.bboxes_curr).cpu()
        if img_meta is not None:
            pred_prev = _bboxes_to_input_space(pred_prev, img_meta).cpu()
            pred_curr = _bboxes_to_input_space(pred_curr, img_meta).cpu()
        scores = pred.scores.cpu()
        pres_prev = pred.presence_prev.cpu()
        pres_curr = pred.presence_curr.cpu()
        query_ids = list(matches.values())
        _draw_pred_boxes(
            left, pred_prev[query_ids],
            [f'p{q} s={scores[q]:.2f} pr={pres_prev[q]:.2f}' for q in query_ids],
            (255, 255, 0))
        _draw_pred_boxes(
            right, pred_curr[query_ids],
            [f'p{q} s={scores[q]:.2f} pr={pres_curr[q]:.2f}' for q in query_ids],
            (255, 255, 0))
    else:
        raise ValueError(f'Unknown visualization view: {view}')

    vis = np.concatenate([left, right], axis=1)
    if save_path is not None:
        mkdir_or_exist(osp.dirname(save_path))
        cv2.imwrite(save_path, vis)
    return vis


@HOOKS.register_module()
class HSMOTPairValVisualizationHook(Hook):
    """Save per-pair GT/pred visualizations during validation."""

    def __init__(self,
                 draw: bool = True,
                 score_thr: float = 0.35,
                 iou_thr: float = 0.5,
                 pres_thr: float = 0.5,
                 out_dir: str = 'val_vis',
                 max_samples: Optional[int] = None,
                 max_samples_per_sequence: Optional[int] = 1,
                 views: Sequence[str] = ('deploy', 'low_score', 'iou_diag')) -> None:
        self.draw = draw
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.pres_thr = pres_thr
        self.out_dir = out_dir
        self.max_samples = max_samples
        self.max_samples_per_sequence = max_samples_per_sequence
        self.views = tuple(views)
        unknown_views = set(self.views) - {'deploy', 'low_score', 'iou_diag'}
        if unknown_views:
            raise ValueError(f'Unknown pair visualization views: {unknown_views}')
        self._save_root: Optional[str] = None
        self._pair_idx = 0
        self._seq_counts = defaultdict(int)

    def before_val_epoch(self, runner: Runner) -> None:
        rank, _ = get_dist_info()
        if rank != 0:
            return
        self._pair_idx = 0
        self._seq_counts.clear()
        self._save_root = osp.join(
            runner.work_dir, self.out_dir, f'iter_{runner.iter:06d}')
        mkdir_or_exist(self._save_root)

    def _should_save_sample(self, meta: dict) -> bool:
        if self.max_samples is not None and self._pair_idx >= self.max_samples:
            return False
        if self.max_samples_per_sequence is None:
            return True
        seq = meta.get('video_id', meta.get('seq_name', 'seq'))
        return self._seq_counts[seq] < self.max_samples_per_sequence

    def after_val_iter(self,
                       runner: Runner,
                       batch_idx: int,
                       data_batch: dict,
                       outputs: Sequence) -> None:
        if not self.draw:
            return
        rank, _ = get_dist_info()
        if rank != 0 or self._save_root is None:
            return

        inputs = data_batch['inputs']
        if isinstance(inputs, torch.Tensor):
            input_list = [inputs[i] for i in range(inputs.shape[0])]
        else:
            input_list = inputs

        for sample, pair_input in zip(outputs, input_list):
            if self.max_samples is not None and self._pair_idx >= self.max_samples:
                return
            if not hasattr(sample, 'pair_gt_instances'):
                continue
            if not hasattr(sample, 'pred_pair_instances'):
                continue

            meta = sample.metainfo
            if not self._should_save_sample(meta):
                continue
            img_shape = tuple(meta.get('img_shape', pair_input.shape[-2:]))
            img_prev, img_curr = _pair_frames_for_vis(pair_input, img_shape)
            img_prev = _to_numpy_image(img_prev)
            img_curr = _to_numpy_image(img_curr)
            meta_line = (
                f'{meta.get("video_id", "seq")} '
                f'prev={meta.get("frame_id_prev", "?")} '
                f'curr={meta.get("frame_id", "?")} '
                f'iter={runner.iter}')
            pair_name = (
                f'{meta.get("video_id", "seq")}_'
                f'{meta.get("frame_id_prev", 0)}_'
                f'{meta.get("frame_id", 0)}')
            for view in self.views:
                suffix = '' if view == 'deploy' else f'_{view}'
                visualize_hsmot_pair_pred_gt(
                    img_prev,
                    img_curr,
                    sample.pair_gt_instances,
                    sample.pred_pair_instances,
                    score_thr=self.score_thr,
                    iou_thr=self.iou_thr,
                    pres_thr=self.pres_thr,
                    save_path=osp.join(
                        self._save_root,
                        f'{self._pair_idx:04d}_{pair_name}{suffix}.jpg'),
                    meta_line=meta_line,
                    img_meta=meta,
                    view=view,
                )
            seq = meta.get('video_id', meta.get('seq_name', 'seq'))
            self._seq_counts[seq] += 1
            self._pair_idx += 1
