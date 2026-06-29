#!/usr/bin/env python3
"""Inspect single-frame overfit predictions against GT boxes."""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import torch
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmrotate.registry import DATASETS, MODELS
from mmrotate.structures.bbox import RotatedBoxes, qbox2rbox
from mmrotate.utils import register_all_modules
from torch.utils.data import DataLoader

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

import projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr  # noqa: F401,E402
from projects.multispec_pair_rotated_rtdetr.tools.run_hsmot_single_overfit_acceptance import (  # noqa: E501
    _build_preprocessor,
    _det_loss_sum,
    _prepare_det_batch,
    _rbox_iou,
    _sync_single_dataset_cfg,
    collate_det_batch,
)


def _to_tensor(boxes) -> torch.Tensor:
    if hasattr(boxes, 'tensor'):
        boxes = boxes.tensor
    if boxes.size(-1) == 8:
        boxes = qbox2rbox(boxes)
    return boxes


def _regularized_gt(gt: InstanceData, angle_cfg: dict) -> torch.Tensor:
    boxes = gt.bboxes.clone()
    if hasattr(boxes, 'regularize_boxes'):
        boxes.regularize_boxes(**angle_cfg)
        return _to_tensor(boxes)
    rboxes = RotatedBoxes(_to_tensor(boxes).clone())
    rboxes.regularize_boxes(**angle_cfg)
    return rboxes.tensor


def _best_iou(pred_boxes: torch.Tensor, gt_box: torch.Tensor) -> tuple[float, int]:
    if len(pred_boxes) == 0:
        return 0.0, -1
    best = 0.0
    best_idx = 0
    for idx, pred_box in enumerate(pred_boxes):
        iou = _rbox_iou(pred_box.cpu(), gt_box.cpu())
        if iou > best:
            best = iou
            best_idx = idx
    return best, best_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='projects/multispec_pair_rotated_rtdetr/configs/o2_rtdetr_r18vd_overfit.py')
    parser.add_argument(
        '--data-root',
        default='/data/users/litianhao01/PairMmot/tmp/hsmot_single_overfit_accept/data')
    parser.add_argument(
        '--checkpoint',
        default='/data/users/litianhao01/PairMmot/tmp/hsmot_single_overfit_accept/work_dir/epoch_1.pth')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-samples', type=int, default=2)
    parser.add_argument('--score-thr', type=float, default=0.35)
    args = parser.parse_args()

    register_all_modules()
    cfg = Config.fromfile(args.config)
    _sync_single_dataset_cfg(cfg, osp.abspath(args.data_root), is_real_layout=True)

    preprocessor = _build_preprocessor(cfg)
    model = MODELS.build(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'load_state missing={len(missing)} unexpected={len(unexpected)}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    preprocessor = preprocessor.to(device)
    model.eval()

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_det_batch)

    for sample_idx, batch in enumerate(loader):
        if sample_idx >= args.max_samples:
            break

        inputs, samples = _prepare_det_batch(
            batch, preprocessor, device, training=False)
        with torch.no_grad():
            model.train()
            train_inputs, train_samples = _prepare_det_batch(
                batch, preprocessor, device, training=True)
            losses = model.loss(train_inputs, train_samples)
            loss_sum = _det_loss_sum(losses)
            model.eval()
            outputs = model.predict(inputs, samples, rescale=False)

        sample = outputs[0]
        gt = sample.gt_instances
        pred = sample.pred_instances
        gt_raw = _to_tensor(gt.bboxes).cpu()
        gt_reg = _regularized_gt(gt, cfg.model.bbox_head.angle_cfg).cpu()
        pred_boxes = _to_tensor(pred.bboxes).cpu()
        pred_scores = pred.scores.cpu()
        pred_labels = pred.labels.cpu()
        gt_labels = gt.labels.cpu()

        print('\n=== sample', sample_idx, '===')
        print('metainfo:', {
            key: sample.metainfo.get(key)
            for key in ('img_id', 'img_path', 'ori_shape', 'img_shape',
                        'pad_shape', 'scale_factor', 'seq_name', 'frame_id')
            if key in sample.metainfo
        })
        print('loss_sum:', round(loss_sum, 6))
        print('num_gt:', len(gt_labels), 'num_pred:', len(pred_scores))
        print('pred score range:',
              float(pred_scores.min()), float(pred_scores.max()))
        print('gt raw range min/max:',
              gt_raw.min(0).values.tolist(), gt_raw.max(0).values.tolist())
        print('gt reg range min/max:',
              gt_reg.min(0).values.tolist(), gt_reg.max(0).values.tolist())
        print('pred range min/max:',
              pred_boxes.min(0).values.tolist(),
              pred_boxes.max(0).values.tolist())

        top_score_ious_raw = []
        top_score_ious_reg = []
        best_ious_reg = []
        best_same_cls_ious_reg = []
        for gi in range(len(gt_labels)):
            label = int(gt_labels[gi])
            same_cls = pred_labels == label
            same_cls_scores = pred_scores.clone()
            same_cls_scores[~same_cls] = -1
            top_idx = int(same_cls_scores.argmax())
            top_score_ious_raw.append(
                _rbox_iou(pred_boxes[top_idx], gt_raw[gi]))
            top_score_ious_reg.append(
                _rbox_iou(pred_boxes[top_idx], gt_reg[gi]))
            best_iou, _ = _best_iou(pred_boxes, gt_reg[gi])
            best_ious_reg.append(best_iou)
            same_cls_boxes = pred_boxes[same_cls]
            best_same_iou, _ = _best_iou(same_cls_boxes, gt_reg[gi])
            best_same_cls_ious_reg.append(best_same_iou)

        def summarize(values: list[float]) -> str:
            if not values:
                return 'empty'
            tensor = torch.tensor(values)
            return (
                f'mean={tensor.mean().item():.4f} '
                f'max={tensor.max().item():.4f} '
                f'>=.5={(tensor >= 0.5).float().mean().item():.4f}')

        print('top-score same-class IoU vs raw GT:', summarize(top_score_ious_raw))
        print('top-score same-class IoU vs regularized GT:',
              summarize(top_score_ious_reg))
        print('best any-class IoU vs regularized GT:', summarize(best_ious_reg))
        print('best same-class IoU vs regularized GT:',
              summarize(best_same_cls_ious_reg))

        for gi in range(min(5, len(gt_labels))):
            label = int(gt_labels[gi])
            same_cls = pred_labels == label
            same_cls_scores = pred_scores.clone()
            same_cls_scores[~same_cls] = -1
            top_idx = int(same_cls_scores.argmax())
            print(
                f'gt#{gi} label={label} raw={gt_raw[gi].tolist()} '
                f'reg={gt_reg[gi].tolist()} '
                f'top_idx={top_idx} score={float(pred_scores[top_idx]):.4f} '
                f'top_box={pred_boxes[top_idx].tolist()} '
                f'iou_reg={top_score_ious_reg[gi]:.4f}')


if __name__ == '__main__':
    main()
