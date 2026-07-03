# Copyright (c) AI4RS. All rights reserved.
"""Validation visualization hook for HSMOT single-frame RT-DETR."""

from __future__ import annotations

import os.path as osp
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
    _rgb_preview,
    _to_numpy_image,
)

from .pair_ap_metric import _to_rbox_tensor
from .pair_val_visualization_hook import (
    _bboxes_to_input_space,
    _match_pred_indices,
)


def _polygons_from_rboxes(bboxes: torch.Tensor) -> np.ndarray:
    if bboxes.numel() == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    qbox = rbox2qbox(bboxes)
    return qbox.reshape(-1, 4, 2).cpu().numpy()


def _draw_boxes(canvas: np.ndarray,
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


def _crop_to_img_shape(frame: torch.Tensor, img_shape: tuple) -> torch.Tensor:
    h, w = int(img_shape[0]), int(img_shape[1])
    return frame[..., :h, :w]


def visualize_hsmot_single_pred_gt(
    img,
    gt: InstanceData,
    pred: InstanceData,
    *,
    score_thr: float = 0.35,
    save_path: Optional[str] = None,
    meta_line: Optional[str] = None,
    img_meta: Optional[dict] = None,
) -> np.ndarray:
    """Draw GT (green) and matched predictions (orange) for one frame."""
    canvas = _rgb_preview(_to_numpy_image(img)).copy()
    num_gt = len(gt.labels) if gt.labels is not None else 0

    gt_boxes = _to_rbox_tensor(gt.bboxes).cpu()
    if img_meta is not None:
        gt_boxes = _bboxes_to_input_space(gt_boxes, img_meta).cpu()
    gt_labels = [
        f'gt{i} c={int(gt.labels[i].item())}' for i in range(num_gt)
    ]
    if gt_boxes.numel() > 0:
        _draw_boxes(canvas, gt_boxes, gt_labels, (0, 255, 0))

    matches = _match_pred_indices(gt, pred, score_thr=score_thr)
    matched = sum(1 for q in matches.values() if q >= 0)
    match_ratio = matched / max(num_gt, 1)

    pred_boxes = _to_rbox_tensor(pred.bboxes).cpu()
    if img_meta is not None:
        pred_boxes = _bboxes_to_input_space(pred_boxes, img_meta).cpu()
    pred_scores = pred.scores.cpu()
    pred_labels = pred.labels.cpu()

    matched_boxes = []
    matched_text = []
    for gi, q in matches.items():
        if q < 0:
            continue
        matched_boxes.append(pred_boxes[q])
        matched_text.append(
            f'q{q} s={pred_scores[q]:.2f} c={int(pred_labels[q].item())}')
    if matched_boxes:
        _draw_boxes(canvas, torch.stack(matched_boxes), matched_text,
                    (0, 165, 255))

    title = meta_line or 'single overfit'
    summary = f'{title} | match={matched}/{num_gt} ({match_ratio:.2f})'
    cv2.putText(canvas, summary, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 2, cv2.LINE_AA)

    if save_path is not None:
        mkdir_or_exist(osp.dirname(save_path))
        cv2.imwrite(save_path, canvas)
    return canvas


@HOOKS.register_module()
class HSMOTSingleValVisualizationHook(Hook):
    """Save per-frame GT/pred visualizations during validation."""

    def __init__(self,
                 draw: bool = True,
                 score_thr: float = 0.35,
                 iou_thr: float = 0.5,
                 out_dir: str = 'val_vis') -> None:
        self.draw = draw
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.out_dir = out_dir
        self._save_root: Optional[str] = None
        self._frame_idx = 0

    def before_val_epoch(self, runner: Runner) -> None:
        rank, _ = get_dist_info()
        if rank != 0:
            return
        self._frame_idx = 0
        self._save_root = osp.join(
            runner.work_dir, self.out_dir, f'iter_{runner.iter:06d}')
        mkdir_or_exist(self._save_root)

    def after_val_iter(self,
                       runner: Runner,
                       batch_idx: int,
                       data_batch: dict,
                       outputs: Sequence) -> None:
        del batch_idx
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

        for sample, frame_input in zip(outputs, input_list):
            if not hasattr(sample, 'gt_instances'):
                continue
            if not hasattr(sample, 'pred_instances'):
                continue

            meta = sample.metainfo
            img_shape = tuple(meta.get('img_shape', frame_input.shape[-2:]))
            frame = _crop_to_img_shape(frame_input, img_shape)
            seq = meta.get('seq_name', meta.get('video_id', 'seq'))
            frame_id = meta.get('frame_id', self._frame_idx + 1)
            meta_line = f'{seq} frame={frame_id} iter={runner.iter}'
            frame_name = f'{seq}_{frame_id:06d}'
            save_path = osp.join(
                self._save_root, f'{self._frame_idx:04d}_{frame_name}.jpg')
            visualize_hsmot_single_pred_gt(
                frame,
                sample.gt_instances,
                sample.pred_instances,
                score_thr=self.score_thr,
                save_path=save_path,
                meta_line=meta_line,
                img_meta=meta,
            )
            self._frame_idx += 1
