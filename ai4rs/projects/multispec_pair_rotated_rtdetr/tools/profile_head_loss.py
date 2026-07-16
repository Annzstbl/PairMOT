#!/usr/bin/env python3
"""Profile PairRotatedRTDETRHead vs RotatedRTDETRHead loss bottlenecks."""

from __future__ import annotations

import os
import sys
import time
from typing import Callable, Dict, List, Tuple

import torch
from mmengine.structures import InstanceData

_AI4RS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

from mmrotate.structures.bbox import RotatedBoxes
from mmrotate.utils import register_all_modules
from mmdet.models.task_modules import FocalLossCost, HungarianAssigner
from projects.rotated_dino.rotated_dino.match_cost import ChamferCost, GDCost
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr import (
    PairHungarianAssigner,
    PairRotatedRTDETRHead,
)
from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETRHead

register_all_modules(init_default_scope=True)

IMG_META = dict(img_shape=(640, 800), scale_factor=(1.0, 1.0, 1.0, 1.0))
ANGLE_FACTOR = 3.141592653589793


def cuda_timer(use_cuda: bool) -> Callable[[], float]:
    if not use_cuda:
        return time.perf_counter
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def _now():
        return start

    return _now


class BenchTimer:
    def __init__(self, use_cuda: bool) -> None:
        self.use_cuda = use_cuda

    def measure(self, fn: Callable[[], None], warmup: int, repeats: int) -> float:
        for _ in range(warmup):
            fn()
        if self.use_cuda:
            torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        if not self.use_cuda:
            t0 = time.perf_counter()
            for _ in range(repeats):
                fn()
            return (time.perf_counter() - t0) / repeats
        start_evt.record()
        for _ in range(repeats):
            fn()
        end_evt.record()
        end_evt.synchronize()
        return start_evt.elapsed_time(end_evt) / (1000.0 * repeats)


def _norm_rbox(cx, cy, w, h, angle=0.0) -> torch.Tensor:
    return torch.tensor(
        [cx / 800, cy / 640, w / 800, h / 640, angle / ANGLE_FACTOR],
        dtype=torch.float32)


def _unnorm_rbox(box: torch.Tensor) -> torch.Tensor:
    factor = torch.tensor([800, 640, 800, 640, ANGLE_FACTOR], dtype=box.dtype)
    return box * factor


def _pair_gt(num_gt: int, device: torch.device) -> InstanceData:
    gt = InstanceData()
    gt.labels = torch.arange(num_gt, device=device, dtype=torch.long) % 3
    prev = []
    curr = []
    for i in range(num_gt):
        cx = 0.2 + 0.6 * (i % 5) / 5
        cy = 0.2 + 0.6 * (i // 5) / max(1, num_gt // 5)
        prev.append(_norm_rbox(cx, cy, 0.08, 0.06))
        curr.append(_norm_rbox(cx + 0.02, cy - 0.01, 0.08, 0.06))
    gt.bboxes_prev = torch.stack([_unnorm_rbox(b) for b in prev]).to(device)
    gt.bboxes_curr = torch.stack([_unnorm_rbox(b) for b in curr]).to(device)
    gt.valid_prev = torch.ones(num_gt, dtype=torch.bool, device=device)
    gt.valid_curr = torch.ones(num_gt, dtype=torch.bool, device=device)
    return gt


def _single_gt(num_gt: int, device: torch.device) -> InstanceData:
    gt = InstanceData()
    gt.labels = torch.arange(num_gt, device=device, dtype=torch.long) % 3
    boxes = []
    for i in range(num_gt):
        cx = 0.2 + 0.6 * (i % 5) / 5
        cy = 0.2 + 0.6 * (i // 5) / max(1, num_gt // 5)
        boxes.append(_unnorm_rbox(_norm_rbox(cx, cy, 0.08, 0.06)))
    gt.bboxes = RotatedBoxes(torch.stack(boxes).to(device))
    return gt


def _build_pair_head(num_layers: int, num_classes: int, embed_dims: int,
                     device: torch.device) -> PairRotatedRTDETRHead:
    assigner = dict(
        type='PairHungarianAssigner',
        match_costs=[
            dict(type='mmdet.FocalLossCost', weight=2.0),
            dict(type='PairChamferCost', side='prev', weight=5.0),
            dict(type='PairChamferCost', side='curr', weight=5.0),
            dict(
                type='PairGDCost',
                side='prev',
                loss_type='kld',
                fun='log1p',
                tau=1,
                sqrt=False,
                weight=2.0),
            dict(
                type='PairGDCost',
                side='curr',
                loss_type='kld',
                fun='log1p',
                tau=1,
                sqrt=False,
                weight=2.0),
            dict(type='PairPresenceBCECost', side='prev', weight=1.0),
            dict(type='PairPresenceBCECost', side='curr', weight=1.0),
        ])
    return PairRotatedRTDETRHead(
        num_classes=num_classes,
        embed_dims=embed_dims,
        num_pred_layer=num_layers,
        angle_cfg=dict(width_longer=True, start_angle=0),
        angle_factor=ANGLE_FACTOR,
        sync_cls_avg_factor=False,
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=5.0),
        loss_iou=dict(
            type='mmrotate.GDLoss',
            loss_type='kld',
            fun='log1p',
            tau=1,
            sqrt=False,
            loss_weight=2.0),
        loss_presence=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        train_cfg=dict(assigner=assigner),
        test_cfg=dict(max_per_img=300),
    ).to(device)


def _build_orig_head(num_layers: int, num_classes: int, embed_dims: int,
                     device: torch.device) -> RotatedRTDETRHead:
    return RotatedRTDETRHead(
        num_classes=num_classes,
        embed_dims=embed_dims,
        num_pred_layer=num_layers,
        angle_cfg=dict(width_longer=True, start_angle=0),
        angle_factor=ANGLE_FACTOR,
        sync_cls_avg_factor=False,
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            varifocal_loss_iou_type='hbox_iou'),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=5.0),
        loss_iou=dict(
            type='mmrotate.GDLoss',
            loss_type='kld',
            fun='log1p',
            tau=1,
            sqrt=False,
            loss_weight=2.0),
        train_cfg=dict(
            assigner=dict(
                type=HungarianAssigner,
                match_costs=[
                    dict(type=FocalLossCost, weight=2.0),
                    dict(type=ChamferCost, weight=5.0, box_format='xywha'),
                    dict(
                        type=GDCost,
                        loss_type='kld',
                        fun='log1p',
                        tau=1,
                        sqrt=False,
                        weight=2.0),
                ])),
        test_cfg=dict(max_per_img=300),
    ).to(device)


def _make_pair_inputs(
    head: PairRotatedRTDETRHead,
    batch_size: int,
    num_queries: int,
    num_layers: int,
    device: torch.device,
):
    hidden = [torch.randn(batch_size, num_queries, head.embed_dims, device=device)
              for _ in range(num_layers)]
    ref_prev = [torch.rand(batch_size, num_queries, 5, device=device)
                for _ in range(num_layers)]
    ref_curr = [torch.rand(batch_size, num_queries, 5, device=device)
                for _ in range(num_layers)]
    return hidden, ref_prev, ref_curr


def _profile_pair_assigner_costs(
    assigner: PairHungarianAssigner,
    num_queries: int,
    num_gt: int,
    device: torch.device,
    timer: BenchTimer,
) -> Dict[str, float]:
    gt = _pair_gt(num_gt, device)
    cls = torch.randn(num_queries, 3, device=device)
    prev = torch.rand(num_queries, 5, device=device)
    curr = torch.rand(num_queries, 5, device=device)
    prev[:, 0:4:2] *= 800
    prev[:, 1:4:2] *= 640
    prev[:, 4] *= ANGLE_FACTOR
    curr[:, 0:4:2] *= 800
    curr[:, 1:4:2] *= 640
    curr[:, 4] *= ANGLE_FACTOR
    pred = InstanceData(
        scores=cls,
        bboxes_prev=prev,
        bboxes_curr=curr,
        presence_prev=torch.randn(num_queries, device=device),
        presence_curr=torch.randn(num_queries, device=device),
    )

    timings: Dict[str, float] = {}

    def _sum_costs():
        total = None
        for cost_fn in assigner.match_costs:
            c = cost_fn(pred, gt, IMG_META)
            total = c if total is None else total + c
        return total

    timings['assigner_all_costs'] = timer.measure(_sum_costs, 3, 20)

    for idx, cost_fn in enumerate(assigner.match_costs):
        name = cost_fn.__class__.__name__
        if hasattr(cost_fn, 'side'):
            name = f'{name}_{cost_fn.side}'

        def _one_cost(fn=cost_fn):
            fn(pred, gt, IMG_META)

        timings[f'cost_{idx}_{name}'] = timer.measure(_one_cost, 3, 20)

    timings['assigner_full'] = timer.measure(
        lambda: assigner.assign(pred, gt, IMG_META), 3, 20)
    return timings


def _profile_orig_assigner_costs(
    head: RotatedRTDETRHead,
    num_queries: int,
    num_gt: int,
    device: torch.device,
    timer: BenchTimer,
) -> Dict[str, float]:
    gt = _single_gt(num_gt, device)
    cls = torch.randn(num_queries, head.num_classes, device=device)
    bbox = torch.rand(num_queries, 5, device=device)
    bbox[:, 0:4:2] *= 800
    bbox[:, 1:4:2] *= 640
    bbox[:, 4] *= ANGLE_FACTOR
    pred = InstanceData(scores=cls, bboxes=bbox)
    assigner = head.assigner
    timings: Dict[str, float] = {}

    def _sum_costs():
        total = None
        for cost_fn in assigner.match_costs:
            c = cost_fn(pred, gt, IMG_META)
            total = c if total is None else total + c
        return total

    timings['assigner_all_costs'] = timer.measure(_sum_costs, 3, 20)
    for idx, cost_fn in enumerate(assigner.match_costs):
        name = cost_fn.__class__.__name__

        def _one_cost(fn=cost_fn):
            fn(pred, gt, IMG_META)

        timings[f'cost_{idx}_{name}'] = timer.measure(_one_cost, 3, 20)
    timings['assigner_full'] = timer.measure(
        lambda: assigner.assign(pred, gt, IMG_META), 3, 20)
    return timings


def _profile_pair_head_breakdown(
    head: PairRotatedRTDETRHead,
    batch_size: int,
    num_queries: int,
    num_gt: int,
    num_layers: int,
    device: torch.device,
    timer: BenchTimer,
) -> Dict[str, float]:
    hidden, ref_prev, ref_curr = _make_pair_inputs(
        head, batch_size, num_queries, num_layers, device)
    gt_list = [_pair_gt(num_gt, device) for _ in range(batch_size)]
    meta_list = [IMG_META for _ in range(batch_size)]

    outs = head.forward(hidden, ref_prev, ref_curr)
    (all_cls, all_pres_prev, all_pres_curr, all_bbox_prev,
     all_bbox_curr) = outs

    layer = 0
    cls_scores = all_cls[layer]
    pres_prev = all_pres_prev[layer]
    pres_curr = all_pres_curr[layer]
    bbox_prev = all_bbox_prev[layer]
    bbox_curr = all_bbox_curr[layer]

    timings: Dict[str, float] = {}

    def _get_targets():
        num_imgs = cls_scores.size(0)
        return head.get_targets(
            [cls_scores[i] for i in range(num_imgs)],
            [pres_prev[i] for i in range(num_imgs)],
            [pres_curr[i] for i in range(num_imgs)],
            [bbox_prev[i] for i in range(num_imgs)],
            [bbox_curr[i] for i in range(num_imgs)],
            gt_list,
            meta_list,
        )

    timings['get_targets_1layer'] = timer.measure(_get_targets, 3, 20)
    timings['loss_by_feat_single_1layer'] = timer.measure(
        lambda: head.loss_by_feat_single(
            cls_scores, pres_prev, pres_curr, bbox_prev, bbox_curr,
            gt_list, meta_list),
        3, 20)

    cached_targets = _get_targets()

    def _loss_only():
        (labels_list, label_weights_list, bbox_prev_targets_list,
         bbox_prev_weights_list, bbox_curr_targets_list,
         bbox_curr_weights_list, pres_prev_targets_list,
         pres_curr_targets_list, pres_weights_list, num_total_pos,
         num_total_neg) = cached_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_prev_targets = torch.cat(bbox_prev_targets_list, 0)
        bbox_prev_weights = torch.cat(bbox_prev_weights_list, 0)
        bbox_curr_targets = torch.cat(bbox_curr_targets_list, 0)
        bbox_curr_weights = torch.cat(bbox_curr_weights_list, 0)
        pres_prev_targets = torch.cat(pres_prev_targets_list, 0)
        pres_curr_targets = torch.cat(pres_curr_targets_list, 0)
        pres_weights = torch.cat(pres_weights_list, 0)
        cls_scores_flat = cls_scores.reshape(-1, head.cls_out_channels)
        cls_avg_factor = max(
            num_total_pos * 1.0 + num_total_neg * head.bg_cls_weight, 1)
        head._loss_cls(
            cls_scores_flat, labels, label_weights,
            bbox_prev.reshape(-1, 5), bbox_curr.reshape(-1, 5),
            bbox_prev_targets, bbox_curr_targets,
            bbox_prev_weights, bbox_curr_weights,
            meta_list, cls_avg_factor)
        num_total_pos_val = max(num_total_pos, 1)
        head.loss_presence(
            pres_prev.reshape(-1), pres_prev_targets, pres_weights,
            avg_factor=num_total_pos_val + num_total_neg)
        head.loss_presence(
            pres_curr.reshape(-1), pres_curr_targets, pres_weights,
            avg_factor=num_total_pos_val + num_total_neg)
        factors = head._build_rescale_factors(meta_list, bbox_prev)
        bbox_prev_flat = bbox_prev.reshape(-1, 5)
        bbox_curr_flat = bbox_curr.reshape(-1, 5)
        head.loss_iou(
            bbox_prev_flat * factors, bbox_prev_targets * factors,
            bbox_prev_weights, avg_factor=num_total_pos_val)
        head.loss_iou(
            bbox_curr_flat * factors, bbox_curr_targets * factors,
            bbox_curr_weights, avg_factor=num_total_pos_val)
        head.loss_bbox(
            bbox_prev_flat, bbox_prev_targets, bbox_prev_weights,
            avg_factor=num_total_pos_val)
        head.loss_bbox(
            bbox_curr_flat, bbox_curr_targets, bbox_curr_weights,
            avg_factor=num_total_pos_val)

    timings['loss_compute_only_1layer'] = timer.measure(_loss_only, 3, 20)
    timings['loss_by_feat_all_layers'] = timer.measure(
        lambda: head.loss_by_feat(
            all_cls, all_pres_prev, all_pres_curr, all_bbox_prev,
            all_bbox_curr,
            batch_pair_gt_instances=gt_list,
            batch_img_metas=meta_list),
        3, 10)
    timings['forward_head'] = timer.measure(
        lambda: head.forward(hidden, ref_prev, ref_curr), 3, 20)
    timings['loss_full'] = timer.measure(
        lambda: head.loss(
            torch.stack(hidden),
            ref_prev,
            ref_curr,
            batch_data_samples=_fake_samples(gt_list, meta_list)),
        3, 10)
    return timings


def _fake_samples(gt_list, meta_list):
    from mmdet.structures import DetDataSample

    samples = []
    for gt, meta in zip(gt_list, meta_list):
        s = DetDataSample()
        s.set_metainfo(meta)
        s.pair_gt_instances = gt
        samples.append(s)
    return samples


def _profile_orig_head_breakdown(
    head: RotatedRTDETRHead,
    batch_size: int,
    num_queries: int,
    num_gt: int,
    num_layers: int,
    device: torch.device,
    timer: BenchTimer,
) -> Dict[str, float]:
    # RT-DETR decoder passes references as [cls_list, bbox_list].
    hidden_list = [
        torch.randn(batch_size, num_queries, head.embed_dims, device=device)
        for _ in range(num_layers)
    ]
    cls_list = [
        torch.randn(batch_size, num_queries, head.num_classes, device=device)
        for _ in range(num_layers)
    ]
    bbox_list = [
        torch.rand(batch_size, num_queries, 5, device=device)
        for _ in range(num_layers)
    ]
    references = [cls_list, bbox_list]
    gt_list = [_single_gt(num_gt, device) for _ in range(batch_size)]
    meta_list = [IMG_META for _ in range(batch_size)]

    all_cls, all_bbox = head.forward(hidden_list, references)
    cls_scores = all_cls[0]
    bbox_preds = all_bbox[0]

    timings: Dict[str, float] = {}

    def _get_targets():
        num_imgs = cls_scores.size(0)
        return head.get_targets(
            [cls_scores[i] for i in range(num_imgs)],
            [bbox_preds[i] for i in range(num_imgs)],
            gt_list,
            meta_list,
        )

    timings['get_targets_1layer'] = timer.measure(_get_targets, 3, 20)
    timings['loss_by_feat_single_1layer'] = timer.measure(
        lambda: head.loss_by_feat_single(
            cls_scores, bbox_preds, gt_list, meta_list),
        3, 20)

    cached_targets = _get_targets()

    def _loss_only():
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cached_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        cls_scores_flat = cls_scores.reshape(-1, head.cls_out_channels)
        cls_avg_factor = max(
            num_total_pos * 1.0 + num_total_neg * head.bg_cls_weight, 1)
        head.loss_cls(
            cls_scores_flat, labels, label_weights, avg_factor=cls_avg_factor)
        num_total_pos_val = max(num_total_pos, 1)
        factors = []
        for img_meta, bbox_pred in zip(meta_list, bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor(
                [img_w, img_h, img_w, img_h,
                 head.angle_factor]).unsqueeze(0).repeat(bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        bbox_preds_flat = bbox_preds.reshape(-1, 5)
        head.loss_iou(
            bbox_preds_flat * factors, bbox_targets * factors, bbox_weights,
            avg_factor=num_total_pos_val)
        head.loss_bbox(
            bbox_preds_flat, bbox_targets, bbox_weights,
            avg_factor=num_total_pos_val)

    timings['loss_compute_only_1layer'] = timer.measure(_loss_only, 3, 20)

    timings['loss_by_feat_matching_only'] = timer.measure(
        lambda: head.loss_by_feat(
            all_cls,
            all_bbox,
            None,
            None,
            gt_list,
            meta_list,
            dn_meta=None),
        3, 10)
    timings['forward_head'] = timer.measure(
        lambda: head.forward(hidden_list, references), 3, 20)
    return timings


def _print_section(title: str, timings: Dict[str, float]) -> None:
    print(f'\n=== {title} ===')
    base = timings.get('loss_by_feat_all_layers') or timings.get(
        'loss_by_feat_matching_only') or timings.get('assigner_full') or 1.0
    for key, value in sorted(timings.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * value / base if base > 0 else 0.0
        print(f'  {key:40s} {value * 1000:8.3f} ms  ({pct:5.1f}% of ref)')


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = device.type == 'cuda'
    timer = BenchTimer(use_cuda)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-queries-pair', type=int, default=50)
    parser.add_argument('--num-queries-orig', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-gt', type=int, default=20)
    parser.add_argument('--num-layers', type=int, default=3)
    args = parser.parse_args()

    batch_size = args.batch_size
    num_queries_pair = args.num_queries_pair
    num_queries_orig = args.num_queries_orig
    num_gt = args.num_gt
    num_layers = args.num_layers
    embed_dims = 256

    print(f'Device: {device}')
    print(
        f'batch_size={batch_size}, num_queries={num_queries_pair}, '
        f'num_gt={num_gt}, num_layers={num_layers}, embed_dims={embed_dims}')

    pair_head = _build_pair_head(num_layers, 3, embed_dims, device)
    orig_head = _build_orig_head(num_layers, 3, embed_dims, device)

    pair_assigner = pair_head.assigner
    pair_assigner_timings = _profile_pair_assigner_costs(
        pair_assigner, num_queries_pair, num_gt, device, timer)
    orig_assigner_timings = _profile_orig_assigner_costs(
        orig_head, num_queries_orig, num_gt, device, timer)

    pair_timings = _profile_pair_head_breakdown(
        pair_head, batch_size, num_queries_pair, num_gt, num_layers, device,
        timer)
    orig_timings = _profile_orig_head_breakdown(
        orig_head, batch_size, num_queries_orig, num_gt, num_layers, device,
        timer)

    _print_section('PairHungarianAssigner (single image)', pair_assigner_timings)
    _print_section('HungarianAssigner original (single image)',
                   orig_assigner_timings)
    _print_section('PairRotatedRTDETRHead loss path', pair_timings)
    _print_section('RotatedRTDETRHead loss path (matching only)', orig_timings)

    pair_total = pair_timings['loss_by_feat_all_layers']
    orig_total = orig_timings['loss_by_feat_matching_only']
    ratio = pair_total / max(orig_total, 1e-9)
    print('\n=== Summary ===')
    print(f'  pair loss_by_feat (all layers): {pair_total * 1000:.3f} ms')
    print(f'  orig loss_by_feat (matching):   {orig_total * 1000:.3f} ms')
    print(f'  slowdown ratio:                 {ratio:.2f}x')
    print(
        f'  pair get_targets / single layer: '
        f'{pair_timings["get_targets_1layer"] * 1000:.3f} ms '
        f'({100 * pair_timings["get_targets_1layer"] / pair_timings["loss_by_feat_single_1layer"]:.1f}% of loss_by_feat_single)')
    print(
        f'  pair loss_compute_only / layer:  '
        f'{pair_timings["loss_compute_only_1layer"] * 1000:.3f} ms')
    print(
        f'  orig get_targets / single layer: '
        f'{orig_timings["get_targets_1layer"] * 1000:.3f} ms '
        f'({100 * orig_timings["get_targets_1layer"] / orig_timings["loss_by_feat_single_1layer"]:.1f}% of loss_by_feat_single)')
    print(
        f'  orig loss_compute_only / layer:   '
        f'{orig_timings["loss_compute_only_1layer"] * 1000:.3f} ms')


if __name__ == '__main__':
    main()
