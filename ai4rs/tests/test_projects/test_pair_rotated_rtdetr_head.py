# Copyright (c) AI4RS. All rights reserved.
"""Unit tests for PairRotatedRTDETRHead / PairHungarianAssigner (M4)."""

import copy
import os.path as osp
import sys
import unittest

import torch
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

from mmrotate.utils import register_all_modules
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr import (
    PairHungarianAssigner,
    PairInstanceData,
    PairRotatedRTDETRHead,
    PairRotatedRTDETRTransformerDecoder,
)
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.pair_cdn_query_generator import (
    PairCdnQueryGenerator,
)
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.rotated_box_utils import (  # noqa: E501
    canonicalize_le180_start0,
    encode_le180_l1,
)

register_all_modules(init_default_scope=True)

IMG_META = dict(img_shape=(640, 800), scale_factor=(1.0, 1.0, 1.0, 1.0))
ANGLE_FACTOR = 3.141592653589793


def _default_train_cfg():
    return dict(
        assigner=dict(
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
            ]))


def _no_presence_train_cfg():
    cfg = copy.deepcopy(_default_train_cfg())
    cfg['assigner']['match_costs'] = [
        cost for cost in cfg['assigner']['match_costs']
        if cost['type'] != 'PairPresenceBCECost'
    ]
    return cfg


def _build_head(num_layers: int = 2,
                num_classes: int = 3,
                embed_dims: int = 32,
                device: torch.device = torch.device('cpu'),
                **head_kwargs) -> PairRotatedRTDETRHead:
    train_cfg = head_kwargs.pop('train_cfg', _default_train_cfg())
    head = PairRotatedRTDETRHead(
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
        train_cfg=train_cfg,
        test_cfg=dict(max_per_img=10),
        **head_kwargs,
    ).to(device)
    return head


def _build_assigner() -> PairHungarianAssigner:
    from mmrotate.registry import TASK_UTILS
    return TASK_UTILS.build(_default_train_cfg()['assigner'])


def _norm_rbox(cx: float, cy: float, w: float, h: float,
               angle: float = 0.0) -> torch.Tensor:
    return torch.tensor(
        [cx / 800, cy / 640, w / 800, h / 640, angle / ANGLE_FACTOR],
        dtype=torch.float32)


def _unnorm_rbox(box: torch.Tensor) -> torch.Tensor:
    factor = torch.tensor([800, 640, 800, 640, ANGLE_FACTOR], dtype=box.dtype)
    return box * factor


def _pair_gt(labels, prev_boxes, curr_boxes, valid_prev, valid_curr):
    gt = InstanceData()
    gt.labels = torch.tensor(labels, dtype=torch.long)
    gt.bboxes_prev = torch.stack(
        [_unnorm_rbox(b) for b in prev_boxes], dim=0)
    gt.bboxes_curr = torch.stack(
        [_unnorm_rbox(b) for b in curr_boxes], dim=0)
    gt.valid_prev = torch.tensor(valid_prev, dtype=torch.bool)
    gt.valid_curr = torch.tensor(valid_curr, dtype=torch.bool)
    return gt


def _pred_instances(cls_logits, prev_boxes, curr_boxes, pres_prev, pres_curr):
    pred = InstanceData()
    pred.scores = cls_logits
    pred.bboxes_prev = torch.stack(
        [_unnorm_rbox(b) for b in prev_boxes], dim=0)
    pred.bboxes_curr = torch.stack(
        [_unnorm_rbox(b) for b in curr_boxes], dim=0)
    pred.presence_prev = pres_prev
    pred.presence_curr = pres_curr
    return pred


class TestPairHungarianAssigner(unittest.TestCase):

    def setUp(self):
        self.assigner = _build_assigner()

    def test_exact_pair_priority_matching(self):
        gt = _pair_gt(
            labels=[0],
            prev_boxes=[_norm_rbox(0.5, 0.5, 0.2, 0.2)],
            curr_boxes=[_norm_rbox(0.52, 0.48, 0.2, 0.2)],
            valid_prev=[True],
            valid_curr=[True],
        )
        good_cls = torch.zeros(1, 3)
        good_cls[0, 0] = 4.0
        bad_cls = torch.zeros(1, 3)
        bad_cls[0, 0] = -4.0
        pred_good = _pred_instances(
            good_cls,
            [_norm_rbox(0.5, 0.5, 0.2, 0.2)],
            [_norm_rbox(0.52, 0.48, 0.2, 0.2)],
            torch.tensor([3.0]),
            torch.tensor([3.0]),
        )
        pred_bad = _pred_instances(
            bad_cls,
            [_norm_rbox(0.1, 0.1, 0.1, 0.1)],
            [_norm_rbox(0.9, 0.9, 0.1, 0.1)],
            torch.tensor([-3.0]),
            torch.tensor([-3.0]),
        )
        pred = InstanceData()
        pred.scores = torch.cat([pred_good.scores, pred_bad.scores], dim=0)
        pred.bboxes_prev = torch.cat(
            [pred_good.bboxes_prev, pred_bad.bboxes_prev], dim=0)
        pred.bboxes_curr = torch.cat(
            [pred_good.bboxes_curr, pred_bad.bboxes_curr], dim=0)
        pred.presence_prev = torch.cat(
            [pred_good.presence_prev, pred_bad.presence_prev], dim=0)
        pred.presence_curr = torch.cat(
            [pred_good.presence_curr, pred_bad.presence_curr], dim=0)

        result = self.assigner.assign(pred, gt, IMG_META)
        matched = torch.nonzero(result.gt_inds > 0, as_tuple=False).squeeze(-1)
        self.assertEqual(matched.numel(), 1)
        self.assertEqual(int(matched.item()), 0)

    def test_swapped_curr_target_increases_cost(self):
        gt_correct = _pair_gt(
            labels=[0],
            prev_boxes=[_norm_rbox(0.5, 0.5, 0.2, 0.2)],
            curr_boxes=[_norm_rbox(0.52, 0.48, 0.2, 0.2)],
            valid_prev=[True],
            valid_curr=[True],
        )
        gt_swapped = _pair_gt(
            labels=[0],
            prev_boxes=[_norm_rbox(0.5, 0.5, 0.2, 0.2)],
            curr_boxes=[_norm_rbox(0.1, 0.1, 0.2, 0.2)],
            valid_prev=[True],
            valid_curr=[True],
        )
        pred = _pred_instances(
            torch.zeros(1, 3),
            [_norm_rbox(0.5, 0.5, 0.2, 0.2)],
            [_norm_rbox(0.52, 0.48, 0.2, 0.2)],
            torch.tensor([0.0]),
            torch.tensor([0.0]),
        )
        costs = []
        for gt in (gt_correct, gt_swapped):
            total = sum(
                c(pred, gt, IMG_META) for c in self.assigner.match_costs)
            costs.append(total[0, 0].item())
        self.assertLess(costs[0], costs[1])

    def test_duplicate_predictions_single_match(self):
        gt = _pair_gt(
            labels=[0],
            prev_boxes=[_norm_rbox(0.5, 0.5, 0.2, 0.2)],
            curr_boxes=[_norm_rbox(0.5, 0.5, 0.2, 0.2)],
            valid_prev=[True],
            valid_curr=[True],
        )
        box = _norm_rbox(0.5, 0.5, 0.2, 0.2)
        pred = _pred_instances(
            torch.zeros(2, 3),
            [box, box.clone()],
            [box.clone(), box.clone()],
            torch.zeros(2),
            torch.zeros(2),
        )
        result = self.assigner.assign(pred, gt, IMG_META)
        self.assertEqual(int((result.gt_inds > 0).sum().item()), 1)


class TestPairRotatedRTDETRHeadLoss(unittest.TestCase):

    def test_le180_l1_treats_swapped_edges_as_same_box(self):
        factor = torch.tensor(
            [[800., 640., 800., 640., ANGLE_FACTOR]])
        physical = torch.tensor([[400., 320., 100., 40., 0.2]])
        swapped = torch.tensor(
            [[400., 320., 40., 100., 0.2 + torch.pi / 2]])
        encoded = encode_le180_l1(physical / factor, factor, 0.05)
        encoded_swapped = encode_le180_l1(swapped / factor, factor, 0.05)
        self.assertTrue(torch.allclose(
            encoded, encoded_swapped, atol=1e-6, rtol=1e-6))
        canonical = canonicalize_le180_start0(swapped)
        self.assertGreaterEqual(canonical[0, 2].item(), canonical[0, 3].item())
        self.assertGreaterEqual(canonical[0, 4].item(), 0.0)
        self.assertLess(canonical[0, 4].item(), torch.pi)

    def test_le180_l1_uses_image_scale_and_fixed_angle_weight(self):
        factor = torch.tensor([
            [800., 640., 800., 640., ANGLE_FACTOR],
            [800., 640., 800., 640., ANGLE_FACTOR],
        ])
        base = torch.tensor([
            [400., 320., 100., 40., 0.2],
            [400., 320., 20., 10., 0.2],
        ])
        shifted_x = base.clone()
        shifted_x[:, 0] += 1
        shifted_y = base.clone()
        shifted_y[:, 1] += 1
        shifted_angle = base.clone()
        shifted_angle[:, 4] += 0.1
        encoded = encode_le180_l1(base / factor, factor, 0.05)
        encoded_x = encode_le180_l1(shifted_x / factor, factor, 0.05)
        encoded_y = encode_le180_l1(shifted_y / factor, factor, 0.05)
        encoded_angle = encode_le180_l1(
            shifted_angle / factor, factor, 0.05)
        self.assertTrue(torch.allclose(
            encoded_x[:, 0] - encoded[:, 0],
            encoded_y[:, 1] - encoded[:, 1], atol=1e-7, rtol=1e-6))
        angle_delta = encoded_angle[:, 4] - encoded[:, 4]
        self.assertTrue(torch.allclose(angle_delta[0], angle_delta[1]))
        self.assertAlmostEqual(
            angle_delta[0].item(), 0.05 * 0.1 / ANGLE_FACTOR, places=7)

    def _run_loss(self, gt, bbox_prev, bbox_curr, require_grad: bool = True):
        head = _build_head(num_layers=1)
        if require_grad:
            bbox_prev = bbox_prev.detach().clone().requires_grad_(True)
            bbox_curr = bbox_curr.detach().clone().requires_grad_(True)
        cls_scores = torch.zeros(1, 1, bbox_prev.size(0), 3)
        cls_scores[0, 0, 0, 0] = 10.0
        presence_prev = torch.zeros(1, 1, bbox_prev.size(0))
        presence_curr = torch.zeros(1, 1, bbox_prev.size(0))
        presence_prev[0, 0, 0] = 3.0
        presence_curr[0, 0, 0] = 3.0
        losses = head.loss_by_feat(
            cls_scores,
            presence_prev,
            presence_curr,
            bbox_prev.unsqueeze(0).unsqueeze(0),
            bbox_curr.unsqueeze(0).unsqueeze(0),
            batch_pair_gt_instances=[gt],
            batch_img_metas=[IMG_META],
        )
        total = sum(v for v in losses.values() if v.requires_grad)
        if require_grad:
            total.backward()
        return head, losses, bbox_prev, bbox_curr

    def test_new_target_only_curr_box_loss(self):
        gt = _pair_gt(
            labels=[0],
            prev_boxes=[_norm_rbox(0.0, 0.0, 0.1, 0.1)],
            curr_boxes=[_norm_rbox(0.5, 0.5, 0.2, 0.2)],
            valid_prev=[False],
            valid_curr=[True],
        )
        prev = _norm_rbox(0.3, 0.3, 0.15, 0.15).unsqueeze(0)
        curr = _norm_rbox(0.52, 0.48, 0.2, 0.2).unsqueeze(0)
        _, losses, grad_prev, grad_curr = self._run_loss(gt, prev, curr)
        self.assertGreater(losses['loss_bbox_curr'].item(), 0.0)
        self.assertEqual(losses['loss_bbox_prev'].item(), 0.0)
        if grad_prev.grad is not None:
            self.assertEqual(grad_prev.grad.abs().sum().item(), 0.0)
        self.assertGreater(grad_curr.grad.abs().sum().item(), 0.0)

    def test_disappear_only_prev_box_loss(self):
        gt = _pair_gt(
            labels=[0],
            prev_boxes=[_norm_rbox(0.5, 0.5, 0.2, 0.2)],
            curr_boxes=[_norm_rbox(0.0, 0.0, 0.1, 0.1)],
            valid_prev=[True],
            valid_curr=[False],
        )
        prev = _norm_rbox(0.52, 0.48, 0.2, 0.2).unsqueeze(0)
        curr = _norm_rbox(0.3, 0.3, 0.15, 0.15).unsqueeze(0)
        _, losses, grad_prev, grad_curr = self._run_loss(gt, prev, curr)
        self.assertGreater(losses['loss_bbox_prev'].item(), 0.0)
        self.assertEqual(losses['loss_bbox_curr'].item(), 0.0)
        self.assertGreater(grad_prev.grad.abs().sum().item(), 0.0)
        if grad_curr.grad is not None:
            self.assertEqual(grad_curr.grad.abs().sum().item(), 0.0)

    def test_missing_box_no_gradient(self):
        gt = _pair_gt(
            labels=[0],
            prev_boxes=[_norm_rbox(0.5, 0.5, 0.2, 0.2)],
            curr_boxes=[_norm_rbox(0.0, 0.0, 0.1, 0.1)],
            valid_prev=[True],
            valid_curr=[False],
        )
        prev = _norm_rbox(0.52, 0.48, 0.2, 0.2).unsqueeze(0)
        curr = _norm_rbox(0.7, 0.7, 0.2, 0.2).unsqueeze(0)
        _, _, _, grad_curr = self._run_loss(gt, prev, curr)
        if grad_curr.grad is None:
            return
        self.assertEqual(grad_curr.grad.abs().sum().item(), 0.0)

    def test_missing_nan_box_is_excluded_from_gd_loss(self):
        head = _build_head(num_layers=1)
        preds = torch.tensor([
            [400.0, 320.0, 80.0, 64.0, 0.0],
            [float('nan'), float('nan'), float('nan'), float('nan'),
             float('nan')],
        ], requires_grad=True)
        targets = torch.tensor([
            [410.0, 315.0, 80.0, 64.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        weights = torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ])

        loss = head._loss_iou_valid(
            preds, targets, weights, avg_factor=1.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(preds.grad).all())
        self.assertEqual(preds.grad[1].abs().sum().item(), 0.0)

    def test_all_missing_nan_boxes_return_finite_graph_zero(self):
        head = _build_head(num_layers=1)
        preds = torch.full((2, 5), float('nan'), requires_grad=True)
        targets = torch.zeros_like(preds)
        weights = torch.zeros_like(preds)

        loss = head._loss_iou_valid(
            preds, targets, weights, avg_factor=1.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(preds.grad).all())
        self.assertEqual(preds.grad.abs().sum().item(), 0.0)

    def test_dual_cls_all_gt_single_visible_targets(self):
        head = _build_head(
            num_layers=1,
            use_presence=False,
            dual_cls=True,
            train_cfg=_no_presence_train_cfg())
        gt = _pair_gt(
            labels=[0],
            prev_boxes=[_norm_rbox(0.0, 0.0, 0.1, 0.1)],
            curr_boxes=[_norm_rbox(0.5, 0.5, 0.2, 0.2)],
            valid_prev=[False],
            valid_curr=[True],
        )
        cls_score = torch.zeros(1, 3)
        bbox_prev = _norm_rbox(0.3, 0.3, 0.15, 0.15).unsqueeze(0)
        bbox_curr = _norm_rbox(0.52, 0.48, 0.2, 0.2).unsqueeze(0)

        targets = head.get_targets_no_presence(
            [cls_score], [bbox_prev], [bbox_curr], [gt], [IMG_META])
        labels_prev, labels_curr = targets[0][0], targets[1][0]
        bbox_prev_weights = targets[4][0]
        bbox_curr_weights = targets[6][0]
        num_total_pos = targets[7]

        self.assertEqual(num_total_pos, 1)
        self.assertEqual(labels_prev[0].item(), head.num_classes)
        self.assertEqual(labels_curr[0].item(), 0)
        self.assertEqual(bbox_prev_weights[0].abs().sum().item(), 0.0)
        self.assertGreater(bbox_curr_weights[0].abs().sum().item(), 0.0)

    def test_pair_dn_targets_use_background_negative_half(self):
        head = _build_head(
            num_layers=1,
            use_presence=False,
            dual_cls=True,
            train_cfg=_no_presence_train_cfg())
        gt = _pair_gt(
            labels=[0, 1],
            prev_boxes=[
                _norm_rbox(0.0, 0.0, 0.1, 0.1),
                _norm_rbox(0.5, 0.5, 0.2, 0.2),
            ],
            curr_boxes=[
                _norm_rbox(0.5, 0.5, 0.2, 0.2),
                _norm_rbox(0.0, 0.0, 0.1, 0.1),
            ],
            valid_prev=[False, True],
            valid_curr=[True, False],
        )
        dn_meta = dict(
            num_denoising_queries=6,
            num_denoising_groups=2,
            max_num_dn_targets=2,
            max_num_negative_dn_targets=1,
            num_negative_dn_targets_per_image=[1])
        targets = head._get_pair_dn_targets(
            [gt], [IMG_META], dn_meta, torch.device('cpu'))
        labels, label_weights = targets[0][0], targets[1][0]
        bbox_prev_weights, bbox_curr_weights = targets[3][0], targets[5][0]
        pres_prev_targets, pres_curr_targets = targets[6][0], targets[7][0]
        num_total_pos, num_total_neg = targets[-2:]

        self.assertEqual(num_total_pos, 4)
        self.assertEqual(num_total_neg, 2)
        self.assertEqual(labels.tolist(), [
            0, 1, head.num_classes,
            0, 1, head.num_classes,
        ])
        self.assertEqual(label_weights.sum().item(), 6.0)
        self.assertEqual(pres_prev_targets[:2].tolist(), [0.0, 1.0])
        self.assertEqual(pres_curr_targets[:2].tolist(), [1.0, 0.0])
        self.assertEqual(bbox_prev_weights[0].abs().sum().item(), 0.0)
        self.assertGreater(bbox_curr_weights[0].abs().sum().item(), 0.0)
        self.assertGreater(bbox_prev_weights[1].abs().sum().item(), 0.0)
        self.assertEqual(bbox_curr_weights[1].abs().sum().item(), 0.0)
        self.assertEqual(bbox_prev_weights[2].abs().sum().item(), 0.0)
        self.assertEqual(bbox_curr_weights[2].abs().sum().item(), 0.0)
        self.assertEqual(bbox_prev_weights[5].abs().sum().item(), 0.0)
        self.assertEqual(bbox_curr_weights[5].abs().sum().item(), 0.0)

    def test_pair_cdn_mask_keeps_positive_and_negative_in_same_group(self):
        gt = _pair_gt(
            labels=[0, 1],
            prev_boxes=[
                _norm_rbox(0.3, 0.3, 0.1, 0.1),
                _norm_rbox(0.7, 0.7, 0.1, 0.1),
            ],
            curr_boxes=[
                _norm_rbox(0.32, 0.3, 0.1, 0.1),
                _norm_rbox(0.72, 0.7, 0.1, 0.1),
            ],
            valid_prev=[True, True],
            valid_curr=[True, True],
        )
        sample = DetDataSample(metainfo=IMG_META)
        sample.pair_gt_instances = gt
        generator = PairCdnQueryGenerator(
            num_classes=3,
            embed_dims=16,
            num_matching_queries=3,
            group_cfg=dict(dynamic=False, num_groups=2))

        _, _, _, mask, query_padding_mask, meta = generator([sample])

        self.assertEqual(meta['num_denoising_queries'], 6)
        self.assertFalse(mask[:3, :3].any())
        self.assertFalse(mask[3:6, 3:6].any())
        self.assertTrue(mask[:3, 3:6].all())
        self.assertTrue(mask[3:6, :3].all())
        self.assertTrue(mask[6:, :6].all())
        self.assertFalse(query_padding_mask.any())

    def test_pair_cdn_easy_hard_positive_layout_and_mask(self):
        gt = _pair_gt(
            labels=[0, 1],
            prev_boxes=[_norm_rbox(.3, .3, .1, .08),
                        _norm_rbox(.7, .7, .12, .08)],
            curr_boxes=[_norm_rbox(.31, .3, .1, .08),
                        _norm_rbox(.71, .7, .12, .08)],
            valid_prev=[True, True], valid_curr=[True, True])
        sample = DetDataSample(metainfo=IMG_META)
        sample.pair_gt_instances = gt
        generator = PairCdnQueryGenerator(
            num_classes=3,
            embed_dims=16,
            num_matching_queries=3,
            dn_target_mode='easy_hard_positive',
            share_pair_noise=False,
            group_cfg=dict(dynamic=False, num_groups=2))

        _, _, _, mask, query_padding_mask, meta = generator([sample])

        self.assertEqual(meta['dn_target_mode'], 'easy_hard_positive')
        self.assertEqual(meta['num_denoising_queries'], 8)
        self.assertEqual(meta['max_num_hard_positive_dn_targets'], 2)
        self.assertEqual(
            meta['num_hard_positive_dn_targets_per_image'], [2])
        for start in (0, 4):
            self.assertFalse(mask[start:start + 2, start:start + 2].any())
            self.assertFalse(mask[start + 2:start + 4,
                                  start + 2:start + 4].any())
            self.assertTrue(mask[start:start + 2,
                                 start + 2:start + 4].all())
            self.assertTrue(mask[start + 2:start + 4,
                                 start:start + 2].all())
        self.assertTrue(mask[:4, 4:8].all())
        self.assertTrue(mask[4:8, :4].all())
        self.assertTrue(mask[8:, :8].all())
        self.assertFalse(query_padding_mask.any())

        easy = generator._sample_unit_noise(
            128, torch.device('cpu'), torch.float32, negative=False,
            positive_hard=False)
        hard = generator._sample_unit_noise(
            128, torch.device('cpu'), torch.float32, negative=False,
            positive_hard=True)
        self.assertTrue(torch.all(easy.abs() < 1.0))
        self.assertTrue(torch.all(
            hard.abs() >= generator.positive_hard_min_magnitude))
        self.assertTrue(torch.all(
            hard.abs() < generator.positive_hard_max_magnitude))

    def test_pair_dn_easy_hard_positive_targets(self):
        head = _build_head(
            num_layers=1,
            use_presence=False,
            dual_cls=True,
            train_cfg=_no_presence_train_cfg())
        gt = _pair_gt(
            labels=[0, 1],
            prev_boxes=[_norm_rbox(0.0, 0.0, 0.1, 0.1),
                        _norm_rbox(.5, .5, .2, .2)],
            curr_boxes=[_norm_rbox(.5, .5, .2, .2),
                        _norm_rbox(0.0, 0.0, 0.1, 0.1)],
            valid_prev=[False, True],
            valid_curr=[True, False])
        dn_meta = dict(
            dn_target_mode='easy_hard_positive',
            num_denoising_queries=4,
            num_denoising_groups=1,
            max_num_dn_targets=2,
            max_num_hard_positive_dn_targets=2,
            num_hard_positive_dn_targets_per_image=[2])

        targets = head._get_pair_dn_targets(
            [gt], [IMG_META], dn_meta, torch.device('cpu'))
        labels, label_weights = targets[0][0], targets[1][0]
        bbox_prev_weights, bbox_curr_weights = targets[3][0], targets[5][0]
        pres_prev_targets, pres_curr_targets = targets[6][0], targets[7][0]
        num_total_pos, num_total_neg = targets[-2:]

        self.assertEqual(labels.tolist(), [0, 1, 0, 1])
        self.assertEqual(label_weights.tolist(), [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(num_total_pos, 4)
        self.assertEqual(num_total_neg, 0)
        self.assertEqual(pres_prev_targets.tolist(), [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(pres_curr_targets.tolist(), [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(bbox_prev_weights[0].abs().sum().item(), 0.0)
        self.assertGreater(bbox_curr_weights[0].abs().sum().item(), 0.0)
        self.assertEqual(bbox_prev_weights[2].abs().sum().item(), 0.0)
        self.assertGreater(bbox_curr_weights[2].abs().sum().item(), 0.0)

    def test_pair_cdn_negative_noise_stays_in_outer_band(self):
        generator = PairCdnQueryGenerator(
            num_classes=3,
            embed_dims=16,
            num_matching_queries=3,
            box_noise_scale=1.0,
            positive_hard_ratio=0.0)
        torch.manual_seed(7)

        positive = generator._sample_unit_noise(
            128, torch.device('cpu'), torch.float32, negative=False)
        negative = generator._sample_unit_noise(
            128, torch.device('cpu'), torch.float32, negative=True)
        self.assertTrue(torch.all(positive.abs() < 1.0))
        self.assertTrue(torch.all(
            negative.abs() >= generator.negative_min_magnitude))
        self.assertTrue(torch.all(
            negative.abs() < generator.negative_max_magnitude))

    def test_pair_cdn_can_share_or_independently_sample_pair_noise(self):
        refs = torch.tensor([
            [0.5, 0.5, 0.2, 0.1, 0.4],
            [0.4, 0.6, 0.16, 0.08, 0.3],
        ])
        valid = torch.ones(2, dtype=torch.bool)
        factor = torch.tensor([1.0, 1.0, 1.0, 1.0, ANGLE_FACTOR])

        shared = PairCdnQueryGenerator(
            num_classes=3, embed_dims=16, num_matching_queries=3,
            box_noise_scale=0.2, positive_hard_ratio=0.0,
            share_pair_noise=True)
        torch.manual_seed(23)
        shared_prev, shared_curr = shared._noisy_pair_refs(
            refs, refs, valid, valid, factor, negative=False)
        self.assertTrue(torch.equal(shared_prev, shared_curr))

        independent = PairCdnQueryGenerator(
            num_classes=3, embed_dims=16, num_matching_queries=3,
            box_noise_scale=0.2, positive_hard_ratio=0.0,
            share_pair_noise=False)
        torch.manual_seed(23)
        independent_prev, independent_curr = independent._noisy_pair_refs(
            refs, refs, valid, valid, factor, negative=False)
        self.assertFalse(torch.equal(independent_prev, independent_curr))

    def test_pair_cdn_padding_mask_and_loss_weights(self):
        gt_full = _pair_gt(
            labels=[0, 1],
            prev_boxes=[_norm_rbox(.3, .3, .1, .08),
                        _norm_rbox(.7, .7, .12, .08)],
            curr_boxes=[_norm_rbox(.31, .3, .1, .08),
                        _norm_rbox(.71, .7, .12, .08)],
            valid_prev=[True, True], valid_curr=[True, True])
        gt_short = _pair_gt(
            labels=[0],
            prev_boxes=[_norm_rbox(.4, .4, .1, .08)],
            curr_boxes=[_norm_rbox(.41, .4, .1, .08)],
            valid_prev=[True], valid_curr=[True])
        samples = []
        for gt in (gt_full, gt_short):
            sample = DetDataSample(metainfo=IMG_META)
            sample.pair_gt_instances = gt
            samples.append(sample)
        generator = PairCdnQueryGenerator(
            num_classes=3, embed_dims=16, num_matching_queries=3,
            group_cfg=dict(dynamic=False, num_groups=1))

        _, _, _, _, query_padding_mask, meta = generator(samples)
        self.assertEqual(meta['num_denoising_queries'], 3)
        self.assertEqual(query_padding_mask[0].tolist(),
                         [False, False, False, False, False, False])
        self.assertEqual(query_padding_mask[1].tolist(),
                         [False, True, False, False, False, False])

        head = _build_head(
            num_layers=1, use_presence=False, dual_cls=True,
            train_cfg=_no_presence_train_cfg())
        targets = head._get_pair_dn_targets(
            [gt_full, gt_short], [IMG_META, IMG_META], meta,
            torch.device('cpu'))
        label_weights = targets[1]
        self.assertEqual(label_weights[0].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(label_weights[1].tolist(), [1.0, 0.0, 1.0])

    def test_pair_cdn_easy_hard_positive_padding_is_unsupervised(self):
        gt_full = _pair_gt(
            labels=[0, 1],
            prev_boxes=[_norm_rbox(.3, .3, .1, .08),
                        _norm_rbox(.7, .7, .12, .08)],
            curr_boxes=[_norm_rbox(.31, .3, .1, .08),
                        _norm_rbox(.71, .7, .12, .08)],
            valid_prev=[True, True], valid_curr=[True, True])
        gt_short = _pair_gt(
            labels=[0],
            prev_boxes=[_norm_rbox(.4, .4, .1, .08)],
            curr_boxes=[_norm_rbox(.41, .4, .1, .08)],
            valid_prev=[True], valid_curr=[True])
        samples = []
        for gt in (gt_full, gt_short):
            sample = DetDataSample(metainfo=IMG_META)
            sample.pair_gt_instances = gt
            samples.append(sample)
        generator = PairCdnQueryGenerator(
            num_classes=3,
            embed_dims=16,
            num_matching_queries=3,
            dn_target_mode='easy_hard_positive',
            share_pair_noise=False,
            group_cfg=dict(dynamic=False, num_groups=1))

        _, _, _, _, query_padding_mask, meta = generator(samples)
        self.assertEqual(meta['num_denoising_queries'], 4)
        self.assertEqual(query_padding_mask[0].tolist(),
                         [False, False, False, False, False, False, False])
        self.assertEqual(query_padding_mask[1].tolist(),
                         [False, True, False, True, False, False, False])

        head = _build_head(
            num_layers=1, use_presence=False, dual_cls=True,
            train_cfg=_no_presence_train_cfg())
        targets = head._get_pair_dn_targets(
            [gt_full, gt_short], [IMG_META, IMG_META], meta,
            torch.device('cpu'))
        self.assertEqual(targets[1][0].tolist(), [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(targets[1][1].tolist(), [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(targets[-2:], (6, 0))

    def test_pair_dn_zero_missing_side_does_not_enter_gd_loss(self):
        head = _build_head(
            num_layers=1,
            use_presence=False,
            dual_cls=True,
            train_cfg=_no_presence_train_cfg())
        gt = _pair_gt(
            labels=[0],
            prev_boxes=[_norm_rbox(0.0, 0.0, 0.0, 0.0)],
            curr_boxes=[_norm_rbox(0.5, 0.5, 0.2, 0.2)],
            valid_prev=[False],
            valid_curr=[True],
        )
        dn_meta = dict(
            num_denoising_queries=2,
            num_denoising_groups=1,
            max_num_dn_targets=1,
            max_num_negative_dn_targets=1,
            num_negative_dn_targets_per_image=[1])
        cls_prev = torch.zeros(1, 2, head.cls_out_channels)
        cls_curr = torch.zeros_like(cls_prev)
        bbox_prev = torch.tensor([[[0.3, 0.3, 0.1, 0.1, 0.0],
                                   [0.4, 0.4, 0.1, 0.1, 0.0]]])
        bbox_curr = torch.tensor([[[0.5, 0.5, 0.2, 0.2, 0.0],
                                   [0.6, 0.6, 0.2, 0.2, 0.0]]])

        losses = head._loss_pair_dn_dual_cls_single(
            cls_prev, cls_curr, bbox_prev, bbox_curr,
            batch_pair_gt_instances=[gt],
            batch_img_metas=[IMG_META],
            dn_meta=dn_meta)

        self.assertTrue(all(torch.isfinite(loss) for loss in losses))
        self.assertEqual(losses[-2].item(), 0.0)
        self.assertGreater(losses[-1].item(), 0.0)

    def test_pair_dn_negative_has_cls_but_no_box_gradient(self):
        head = _build_head(
            num_layers=1,
            use_presence=False,
            dual_cls=True,
            train_cfg=_no_presence_train_cfg())
        gt = _pair_gt(
            labels=[0],
            prev_boxes=[_norm_rbox(0.4, 0.4, 0.2, 0.2)],
            curr_boxes=[_norm_rbox(0.5, 0.5, 0.2, 0.2)],
            valid_prev=[True],
            valid_curr=[True],
        )
        dn_meta = dict(
            num_denoising_queries=2,
            num_denoising_groups=1,
            max_num_dn_targets=1,
            max_num_negative_dn_targets=1,
            num_negative_dn_targets_per_image=[1])
        cls_prev = torch.zeros(
            1, 2, head.cls_out_channels, requires_grad=True)
        cls_curr = torch.zeros_like(cls_prev, requires_grad=True)
        bbox_prev = torch.tensor(
            [[[0.42, 0.42, 0.2, 0.2, 0.0],
              [0.8, 0.8, 0.1, 0.1, 0.0]]], requires_grad=True)
        bbox_curr = torch.tensor(
            [[[0.52, 0.52, 0.2, 0.2, 0.0],
              [0.2, 0.2, 0.1, 0.1, 0.0]]], requires_grad=True)

        losses = head._loss_pair_dn_dual_cls_single(
            cls_prev, cls_curr, bbox_prev, bbox_curr,
            batch_pair_gt_instances=[gt],
            batch_img_metas=[IMG_META],
            dn_meta=dn_meta)
        sum(losses).backward()

        self.assertGreater(cls_prev.grad[0, 1].abs().sum().item(), 0.0)
        self.assertGreater(cls_curr.grad[0, 1].abs().sum().item(), 0.0)
        self.assertGreater(bbox_prev.grad[0, 0].abs().sum().item(), 0.0)
        self.assertGreater(bbox_curr.grad[0, 0].abs().sum().item(), 0.0)
        self.assertEqual(bbox_prev.grad[0, 1].abs().sum().item(), 0.0)
        self.assertEqual(bbox_curr.grad[0, 1].abs().sum().item(), 0.0)


class TestPairRotatedRTDETRHeadForward(unittest.TestCase):

    def test_terminal_pair_common_cls_residual_requires_dual_cls(self):
        with self.assertRaisesRegex(ValueError, 'requires dual_cls'):
            _build_head(
                num_layers=3,
                terminal_pair_common_cls_residual=True)

    def test_terminal_pair_common_cls_residual_preserves_parent_and_detail(
            self):
        head = _build_head(
            num_layers=3,
            use_presence=False,
            dual_cls=True,
            terminal_pair_common_cls_residual=True,
            train_cfg=_no_presence_train_cfg())
        head.init_weights()
        hidden = [torch.randn(1, 5, 32) for _ in range(2)]
        refs = [torch.rand(1, 5, 5) for _ in range(2)]
        dn_meta = dict(num_denoising_queries=2)

        cls_prev, cls_curr, _, _ = head.forward(
            hidden, refs, refs, dn_meta=dn_meta)
        parent_prev = torch.stack([
            head.cls_branches[layer_id](state)
            for layer_id, state in enumerate(hidden)
        ])
        parent_curr = torch.stack([
            head.cls_branches_curr[layer_id](state)
            for layer_id, state in enumerate(hidden)
        ])
        self.assertTrue(torch.equal(cls_prev, parent_prev))
        self.assertTrue(torch.equal(cls_curr, parent_curr))

        with torch.no_grad():
            branch = head.terminal_pair_common_cls_residual_branch
            branch.weight.fill_(0.05)
            branch.bias.fill_(0.1)
        changed_prev, changed_curr, _, _ = head.forward(
            hidden, refs, refs, dn_meta=dn_meta)

        self.assertTrue(torch.equal(changed_prev[0], parent_prev[0]))
        self.assertTrue(torch.equal(changed_curr[0], parent_curr[0]))
        self.assertTrue(torch.equal(
            changed_prev[-1, :, :2], parent_prev[-1, :, :2]))
        self.assertTrue(torch.equal(
            changed_curr[-1, :, :2], parent_curr[-1, :, :2]))
        self.assertGreater(
            (changed_prev[-1, :, 2:] - parent_prev[-1, :, 2:])
            .abs().sum().item(), 0.0)
        parent_detail = parent_curr[-1, :, 2:] - parent_prev[-1, :, 2:]
        changed_detail = (
            changed_curr[-1, :, 2:] - changed_prev[-1, :, 2:])
        torch.testing.assert_close(
            changed_detail, parent_detail, rtol=1e-6, atol=1e-6)

        (changed_prev[-1, :, 2:].sum() +
         changed_curr[-1, :, 2:].sum()).backward()
        self.assertGreater(
            head.terminal_pair_common_cls_residual_branch.weight.grad
            .abs().sum().item(), 0.0)

    def test_terminal_pair_common_and_encoder_cls_residual_are_exclusive(self):
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_head(
                num_layers=3,
                use_presence=False,
                dual_cls=True,
                terminal_encoder_cls_residual=True,
                terminal_pair_common_cls_residual=True,
                train_cfg=_no_presence_train_cfg())

    def test_terminal_pair_common_objectness_requires_dual_cls(self):
        with self.assertRaisesRegex(ValueError, 'requires dual_cls'):
            _build_head(
                num_layers=3,
                terminal_pair_common_objectness_residual=True)

    def test_terminal_pair_common_objectness_preserves_all_margins(self):
        head = _build_head(
            num_layers=3,
            use_presence=False,
            dual_cls=True,
            terminal_pair_common_objectness_residual=True,
            train_cfg=_no_presence_train_cfg())
        head.init_weights()
        hidden = [torch.randn(1, 5, 32) for _ in range(2)]
        refs = [torch.rand(1, 5, 5) for _ in range(2)]
        dn_meta = dict(num_denoising_queries=2)

        parent_prev, parent_curr, _, _ = head.forward(
            hidden, refs, refs, dn_meta=dn_meta)
        with torch.no_grad():
            branch = head.terminal_pair_common_objectness_residual_branch
            branch.weight.fill_(0.05)
            branch.bias.fill_(0.1)
        changed_prev, changed_curr, _, _ = head.forward(
            hidden, refs, refs, dn_meta=dn_meta)

        self.assertTrue(torch.equal(changed_prev[0], parent_prev[0]))
        self.assertTrue(torch.equal(changed_curr[0], parent_curr[0]))
        self.assertTrue(torch.equal(
            changed_prev[-1, :, :2], parent_prev[-1, :, :2]))
        self.assertTrue(torch.equal(
            changed_curr[-1, :, :2], parent_curr[-1, :, :2]))
        self.assertGreater(
            (changed_prev[-1, :, 2:] - parent_prev[-1, :, 2:])
            .abs().sum().item(), 0.0)

        parent_frame_detail = (
            parent_curr[-1, :, 2:] - parent_prev[-1, :, 2:])
        changed_frame_detail = (
            changed_curr[-1, :, 2:] - changed_prev[-1, :, 2:])
        torch.testing.assert_close(
            changed_frame_detail, parent_frame_detail,
            rtol=1e-6, atol=1e-6)

        parent_class_margin = (
            parent_prev[-1, :, 2:]
            - parent_prev[-1, :, 2:, :1])
        changed_class_margin = (
            changed_prev[-1, :, 2:]
            - changed_prev[-1, :, 2:, :1])
        torch.testing.assert_close(
            changed_class_margin, parent_class_margin,
            rtol=1e-6, atol=1e-6)

        (changed_prev[-1, :, 2:].sum()
         + changed_curr[-1, :, 2:].sum()).backward()
        self.assertGreater(
            head.terminal_pair_common_objectness_residual_branch.weight.grad
            .abs().sum().item(), 0.0)

    def test_terminal_pair_common_modes_are_exclusive(self):
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_head(
                num_layers=3,
                use_presence=False,
                dual_cls=True,
                terminal_pair_common_cls_residual=True,
                terminal_pair_common_objectness_residual=True,
                train_cfg=_no_presence_train_cfg())

    def test_terminal_pair_differential_objectness_requires_dual_cls(self):
        with self.assertRaisesRegex(ValueError, 'requires dual_cls'):
            _build_head(
                num_layers=3,
                terminal_pair_differential_objectness_residual=True)

    def test_terminal_pair_differential_objectness_preserves_pair_mean(self):
        head = _build_head(
            num_layers=3,
            use_presence=False,
            dual_cls=True,
            terminal_pair_differential_objectness_residual=True,
            train_cfg=_no_presence_train_cfg())
        head.init_weights()
        hidden = [torch.randn(1, 5, 32) for _ in range(2)]
        refs = [torch.rand(1, 5, 5) for _ in range(2)]
        dn_meta = dict(num_denoising_queries=2)

        parent_prev, parent_curr, _, _ = head.forward(
            hidden, refs, refs, dn_meta=dn_meta)
        with torch.no_grad():
            branch = (
                head.terminal_pair_differential_objectness_residual_branch)
            branch.weight.fill_(0.05)
            branch.bias.fill_(0.1)
        changed_prev, changed_curr, _, _ = head.forward(
            hidden, refs, refs, dn_meta=dn_meta)

        self.assertTrue(torch.equal(changed_prev[0], parent_prev[0]))
        self.assertTrue(torch.equal(changed_curr[0], parent_curr[0]))
        self.assertTrue(torch.equal(
            changed_prev[-1, :, :2], parent_prev[-1, :, :2]))
        self.assertTrue(torch.equal(
            changed_curr[-1, :, :2], parent_curr[-1, :, :2]))
        self.assertGreater(
            (changed_prev[-1, :, 2:] - parent_prev[-1, :, 2:])
            .abs().sum().item(), 0.0)

        parent_pair_mean = 0.5 * (
            parent_prev[-1, :, 2:] + parent_curr[-1, :, 2:])
        changed_pair_mean = 0.5 * (
            changed_prev[-1, :, 2:] + changed_curr[-1, :, 2:])
        torch.testing.assert_close(
            changed_pair_mean, parent_pair_mean,
            rtol=1e-6, atol=1e-6)

        parent_prev_margin = (
            parent_prev[-1, :, 2:]
            - parent_prev[-1, :, 2:, :1])
        changed_prev_margin = (
            changed_prev[-1, :, 2:]
            - changed_prev[-1, :, 2:, :1])
        parent_curr_margin = (
            parent_curr[-1, :, 2:]
            - parent_curr[-1, :, 2:, :1])
        changed_curr_margin = (
            changed_curr[-1, :, 2:]
            - changed_curr[-1, :, 2:, :1])
        torch.testing.assert_close(
            changed_prev_margin, parent_prev_margin,
            rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            changed_curr_margin, parent_curr_margin,
            rtol=1e-6, atol=1e-6)

        parent_frame_detail = (
            parent_curr[-1, :, 2:] - parent_prev[-1, :, 2:])
        changed_frame_detail = (
            changed_curr[-1, :, 2:] - changed_prev[-1, :, 2:])
        self.assertGreater(
            (changed_frame_detail - parent_frame_detail)
            .abs().sum().item(), 0.0)

        changed_curr[-1, :, 2:].sum().backward()
        self.assertGreater(
            head.terminal_pair_differential_objectness_residual_branch.
            weight.grad.abs().sum().item(), 0.0)

    def test_terminal_pair_differential_mode_is_exclusive(self):
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_head(
                num_layers=3,
                use_presence=False,
                dual_cls=True,
                terminal_pair_common_objectness_residual=True,
                terminal_pair_differential_objectness_residual=True,
                train_cfg=_no_presence_train_cfg())

    def test_terminal_encoder_cls_residual_requires_dual_cls(self):
        with self.assertRaisesRegex(ValueError, 'requires dual_cls'):
            _build_head(
                num_layers=3, terminal_encoder_cls_residual=True)

    def test_terminal_encoder_cls_residual_is_final_only_and_dn_isolated(self):
        head = _build_head(
            num_layers=3,
            use_presence=False,
            dual_cls=True,
            terminal_encoder_cls_residual=True,
            train_cfg=_no_presence_train_cfg())
        head.init_weights()
        hidden_prev = [torch.randn(1, 5, 32) for _ in range(2)]
        hidden_curr = [torch.randn(1, 5, 32) for _ in range(2)]
        refs = [torch.rand(1, 5, 5) for _ in range(2)]
        initial_prev = torch.randn(1, 3, 3, requires_grad=True)
        initial_curr = torch.randn(1, 3, 3, requires_grad=True)
        dn_meta = dict(num_denoising_queries=2)

        cls_prev, cls_curr, _, _ = head.forward(
            hidden_prev,
            refs,
            refs,
            hidden_states_prev=hidden_prev,
            hidden_states_curr=hidden_curr,
            initial_cls_prev=initial_prev,
            initial_cls_curr=initial_curr,
            dn_meta=dn_meta)

        self.assertTrue(torch.equal(
            cls_prev[0], head.cls_branches[0](hidden_prev[0])))
        self.assertTrue(torch.equal(
            cls_curr[0], head.cls_branches_curr[0](hidden_curr[0])))
        self.assertTrue(torch.equal(
            cls_prev[-1, :, :2],
            head.cls_branches[1](hidden_prev[1][:, :2])))
        self.assertTrue(torch.equal(
            cls_curr[-1, :, :2],
            head.cls_branches_curr[1](hidden_curr[1][:, :2])))
        self.assertTrue(torch.equal(
            cls_prev[-1, :, 2:], initial_prev.detach()))
        self.assertTrue(torch.equal(
            cls_curr[-1, :, 2:], initial_curr.detach()))
        self.assertEqual(torch.count_nonzero(
            head.terminal_encoder_cls_residual_prev.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(
            head.terminal_encoder_cls_residual_curr.weight).item(), 0)

        (cls_prev[-1].sum() + cls_curr[-1].sum()).backward()
        self.assertIsNone(initial_prev.grad)
        self.assertIsNone(initial_curr.grad)
        self.assertGreater(
            head.terminal_encoder_cls_residual_prev.weight.grad
            .abs().sum().item(), 0.0)
        self.assertGreater(
            head.terminal_encoder_cls_residual_curr.weight.grad
            .abs().sum().item(), 0.0)
        self.assertGreater(
            head.cls_branches[1].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            head.cls_branches_curr[1].weight.grad.abs().sum().item(), 0.0)

    def test_terminal_and_iterative_cls_residual_are_exclusive(self):
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_head(
                num_layers=3,
                use_presence=False,
                dual_cls=True,
                iterative_cls_residual=True,
                terminal_encoder_cls_residual=True,
                train_cfg=_no_presence_train_cfg())

    def test_iterative_cls_residual_requires_dual_cls(self):
        with self.assertRaisesRegex(ValueError, 'requires dual_cls'):
            _build_head(
                num_layers=3, iterative_cls_residual=True)

    def test_iterative_cls_residual_zero_init_and_dn_alignment(self):
        head = _build_head(
            num_layers=3,
            use_presence=False,
            dual_cls=True,
            iterative_cls_residual=True,
            train_cfg=_no_presence_train_cfg())
        head.init_weights()
        hidden = [torch.randn(1, 5, 32) for _ in range(2)]
        refs_prev = [torch.rand(1, 5, 5) for _ in range(2)]
        refs_curr = [torch.rand(1, 5, 5) for _ in range(2)]
        initial_prev = torch.randn(1, 3, 3, requires_grad=True)
        initial_curr = torch.randn(1, 3, 3, requires_grad=True)
        dn_meta = dict(num_denoising_queries=2)

        cls_prev, cls_curr, _, _ = head.forward(
            hidden,
            refs_prev,
            refs_curr,
            initial_cls_prev=initial_prev,
            initial_cls_curr=initial_curr,
            dn_meta=dn_meta)

        expected_prev = torch.cat((torch.zeros(1, 2, 3),
                                   initial_prev.detach()), dim=1)
        expected_curr = torch.cat((torch.zeros(1, 2, 3),
                                   initial_curr.detach()), dim=1)
        self.assertTrue(torch.equal(cls_prev[0], expected_prev))
        self.assertTrue(torch.equal(cls_prev[1], expected_prev))
        self.assertTrue(torch.equal(cls_curr[0], expected_curr))
        self.assertTrue(torch.equal(cls_curr[1], expected_curr))
        self.assertTrue(all(
            not parameter.requires_grad
            for branch in head.cls_branches[:2]
            for parameter in branch.parameters()))
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in head.cls_branches[2].parameters()))

    def test_iterative_cls_residual_detaches_between_layers(self):
        head = _build_head(
            num_layers=3,
            use_presence=False,
            dual_cls=True,
            iterative_cls_residual=True,
            train_cfg=_no_presence_train_cfg())
        head.init_weights()
        with torch.no_grad():
            for branch in head.iterative_cls_residual_branches_prev:
                branch.weight.fill_(0.1)
        hidden = [torch.randn(1, 2, 32) for _ in range(2)]
        refs = [torch.rand(1, 2, 5) for _ in range(2)]
        initial = torch.randn(1, 2, 3, requires_grad=True)
        cls_prev, _, _, _ = head.forward(
            hidden,
            refs,
            refs,
            initial_cls_prev=initial,
            initial_cls_curr=initial)
        cls_prev[-1].sum().backward()
        self.assertIsNone(initial.grad)
        first_grad = (
            head.iterative_cls_residual_branches_prev[0].weight.grad)
        self.assertTrue(
            first_grad is None or torch.count_nonzero(first_grad).item() == 0)
        self.assertIsNotNone(
            head.iterative_cls_residual_branches_prev[1].weight.grad)

    def test_iterative_cls_residual_can_propagate_between_layers(self):
        head = _build_head(
            num_layers=3,
            use_presence=False,
            dual_cls=True,
            iterative_cls_residual=True,
            iterative_cls_detach_between_layers=False,
            train_cfg=_no_presence_train_cfg())
        head.init_weights()
        hidden = [torch.randn(1, 2, 32) for _ in range(2)]
        refs = [torch.rand(1, 2, 5) for _ in range(2)]
        initial = torch.randn(1, 2, 3, requires_grad=True)
        cls_prev, _, _, _ = head.forward(
            hidden,
            refs,
            refs,
            initial_cls_prev=initial,
            initial_cls_curr=initial)
        cls_prev[-1].sum().backward()
        self.assertIsNone(initial.grad)
        self.assertGreater(
            head.iterative_cls_residual_branches_prev[0].weight.grad
            .abs().sum().item(), 0.0)
        self.assertGreater(
            head.iterative_cls_residual_branches_prev[1].weight.grad
            .abs().sum().item(), 0.0)

    def test_iterative_cls_residual_isolates_dn_absolute_classifier(self):
        head = _build_head(
            num_layers=3,
            use_presence=False,
            dual_cls=True,
            iterative_cls_residual=True,
            iterative_cls_dn_absolute=True,
            train_cfg=_no_presence_train_cfg())
        head.init_weights()
        hidden = [torch.randn(1, 5, 32) for _ in range(2)]
        refs = [torch.rand(1, 5, 5) for _ in range(2)]
        initial_prev = torch.randn(1, 3, 3, requires_grad=True)
        initial_curr = torch.randn(1, 3, 3, requires_grad=True)
        dn_meta = dict(num_denoising_queries=2)

        cls_prev, cls_curr, _, _ = head.forward(
            hidden,
            refs,
            refs,
            initial_cls_prev=initial_prev,
            initial_cls_curr=initial_curr,
            dn_meta=dn_meta)

        for layer_id in range(2):
            expected_dn_prev = head.cls_branches[layer_id](
                hidden[layer_id][:, :2])
            expected_dn_curr = head.cls_branches_curr[layer_id](
                hidden[layer_id][:, :2])
            self.assertTrue(torch.equal(
                cls_prev[layer_id, :, :2], expected_dn_prev))
            self.assertTrue(torch.equal(
                cls_curr[layer_id, :, :2], expected_dn_curr))
            self.assertTrue(torch.equal(
                cls_prev[layer_id, :, 2:], initial_prev.detach()))
            self.assertTrue(torch.equal(
                cls_curr[layer_id, :, 2:], initial_curr.detach()))
        self.assertTrue(all(
            parameter.requires_grad
            for branch in head.cls_branches[:2]
            for parameter in branch.parameters()))

        (cls_prev[-1, :, :2].sum() +
         cls_curr[-1, :, :2].sum()).backward()
        self.assertGreater(
            head.cls_branches[1].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            head.cls_branches_curr[1].weight.grad.abs().sum().item(), 0.0)

    def test_iterative_cls_dn_absolute_requires_iterative_mode(self):
        with self.assertRaisesRegex(ValueError, 'requires'):
            _build_head(
                num_layers=3,
                use_presence=False,
                dual_cls=True,
                iterative_cls_dn_absolute=True,
                train_cfg=_no_presence_train_cfg())

    def test_pair_shared_objectness_requires_dn_isolated_iterative_mode(self):
        with self.assertRaisesRegex(ValueError, 'iterative_cls_residual'):
            _build_head(
                num_layers=3,
                use_presence=False,
                dual_cls=True,
                iterative_cls_pair_shared_objectness=True,
                train_cfg=_no_presence_train_cfg())
        with self.assertRaisesRegex(ValueError, 'dn_absolute'):
            _build_head(
                num_layers=3,
                use_presence=False,
                dual_cls=True,
                iterative_cls_residual=True,
                iterative_cls_pair_shared_objectness=True,
                train_cfg=_no_presence_train_cfg())

    def test_pair_shared_objectness_preserves_margins_and_isolates_dn(self):
        head = _build_head(
            num_layers=3,
            use_presence=False,
            dual_cls=True,
            iterative_cls_residual=True,
            iterative_cls_dn_absolute=True,
            iterative_cls_pair_shared_objectness=True,
            iterative_cls_detach_between_layers=False,
            train_cfg=_no_presence_train_cfg())
        head.init_weights()
        with torch.no_grad():
            prev_branch = head.iterative_cls_residual_branches_prev[0]
            curr_branch = head.iterative_cls_residual_branches_curr[0]
            prev_branch.weight.zero_()
            curr_branch.weight.zero_()
            prev_branch.bias.copy_(torch.tensor([1.0, 2.0, 4.0]))
            curr_branch.bias.copy_(torch.tensor([-2.0, 1.0, 5.0]))

        hidden_prev = torch.randn(1, 4, 32)
        hidden_curr = torch.randn(1, 4, 32)
        running_prev = torch.zeros(1, 4, 3)
        running_curr = torch.zeros(1, 4, 3)
        cls_prev, cls_curr, _, _ = head._iterative_cls_pair_layer(
            running_prev,
            running_curr,
            hidden_prev,
            hidden_curr,
            prev_branch,
            curr_branch,
            head.cls_branches[0],
            head.cls_branches_curr[0],
            num_dn=2)

        self.assertTrue(torch.equal(
            cls_prev[:, :2], head.cls_branches[0](hidden_prev[:, :2])))
        self.assertTrue(torch.equal(
            cls_curr[:, :2],
            head.cls_branches_curr[0](hidden_curr[:, :2])))
        prev_normal = cls_prev[:, 2:]
        curr_normal = cls_curr[:, 2:]
        self.assertTrue(torch.allclose(
            prev_normal[..., 0] - prev_normal[..., 2],
            torch.full((1, 2), -3.0)))
        self.assertTrue(torch.allclose(
            curr_normal[..., 0] - curr_normal[..., 2],
            torch.full((1, 2), -7.0)))
        self.assertTrue(torch.allclose(
            prev_normal.mean(dim=-1), curr_normal.mean(dim=-1)))

    def test_pair_shared_objectness_couples_gradients_between_frames(self):
        head = _build_head(
            num_layers=3,
            use_presence=False,
            dual_cls=True,
            iterative_cls_residual=True,
            iterative_cls_dn_absolute=True,
            iterative_cls_pair_shared_objectness=True,
            iterative_cls_detach_between_layers=False,
            train_cfg=_no_presence_train_cfg())
        head.init_weights()
        hidden_prev = [torch.randn(1, 3, 32) for _ in range(2)]
        hidden_curr = [torch.randn(1, 3, 32) for _ in range(2)]
        refs = [torch.rand(1, 3, 5) for _ in range(2)]
        initial_prev = torch.randn(1, 3, 3)
        initial_curr = torch.randn(1, 3, 3)
        cls_prev, _, _, _ = head.forward(
            hidden_prev,
            refs,
            refs,
            hidden_states_prev=hidden_prev,
            hidden_states_curr=hidden_curr,
            initial_cls_prev=initial_prev,
            initial_cls_curr=initial_curr)
        cls_prev[-1].sum().backward()
        self.assertGreater(
            head.iterative_cls_residual_branches_curr[0].weight.grad
            .abs().sum().item(), 0.0)
        self.assertGreater(
            head.iterative_cls_residual_branches_curr[1].weight.grad
            .abs().sum().item(), 0.0)

    def test_iterative_cls_residual_rejects_query_mismatch(self):
        head = _build_head(
            num_layers=2,
            use_presence=False,
            dual_cls=True,
            iterative_cls_residual=True,
            train_cfg=_no_presence_train_cfg())
        with self.assertRaisesRegex(ValueError, 'alignment mismatch'):
            head.forward(
                [torch.randn(1, 4, 32)],
                [torch.rand(1, 4, 5)],
                [torch.rand(1, 4, 5)],
                initial_cls_prev=torch.randn(1, 3, 3),
                initial_cls_curr=torch.randn(1, 3, 3))

    def test_reg_branches_curr_synced_with_prev(self):
        head = _build_head(num_layers=2, embed_dims=32)
        head.init_weights()
        prev_only_state = {
            key: value.clone()
            for key, value in head.state_dict().items()
            if key.startswith('reg_branches.')
            and not key.startswith('reg_branches_curr.')
        }
        for value in prev_only_state.values():
            value.normal_(mean=0.05, std=0.02)
        head.load_state_dict(prev_only_state, strict=False)
        layer_input = torch.randn(1, 4, 32)
        for lid in range(2):
            tmp_prev = head.reg_branches[lid](layer_input)
            tmp_curr = head.reg_branches_curr[lid](layer_input)
            self.assertTrue(torch.allclose(tmp_prev, tmp_curr))
        self.assertGreater(tmp_curr.abs().max().item(), 0.0)

    def test_forward_output_shapes(self):
        head = _build_head(num_layers=2)
        hidden = [
            torch.randn(2, 4, 32),
            torch.randn(2, 4, 32),
        ]
        ref_prev = [
            torch.rand(2, 4, 5),
            torch.rand(2, 4, 5),
        ]
        ref_curr = [
            torch.rand(2, 4, 5),
            torch.rand(2, 4, 5),
        ]
        cls, pres_p, pres_c, bbox_p, bbox_c = head.forward(
            hidden, ref_prev, ref_curr)
        self.assertEqual(cls.shape, (2, 2, 4, 3))
        self.assertEqual(pres_p.shape, (2, 2, 4))
        self.assertEqual(pres_c.shape, (2, 2, 4))
        self.assertEqual(bbox_p.shape, (2, 2, 4, 5))
        self.assertEqual(bbox_c.shape, (2, 2, 4, 5))

    def test_predict_returns_pair_instance_data(self):
        head = _build_head(num_layers=1)
        cls = torch.randn(1, 1, 2, 3)
        pres_p = torch.randn(1, 1, 2)
        pres_c = torch.randn(1, 1, 2)
        bbox_p = torch.rand(1, 1, 2, 5)
        bbox_c = torch.rand(1, 1, 2, 5)
        results = head.predict_by_feat(
            cls, pres_p, pres_c, bbox_p, bbox_c, batch_img_metas=[IMG_META])
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], PairInstanceData)
        self.assertEqual(results[0].bboxes_prev.shape[-1], 5)
        self.assertEqual(results[0].bboxes_curr.shape[-1], 5)
        for boxes in (results[0].bboxes_prev, results[0].bboxes_curr):
            self.assertTrue(torch.all(boxes[:, 2] >= boxes[:, 3]))
            self.assertTrue(torch.all(boxes[:, 4] >= 0))
            self.assertTrue(torch.all(boxes[:, 4] < torch.pi))

    def test_dual_cls_common_label_uses_stronger_visible_side(self):
        head = _build_head(
            num_layers=1,
            use_presence=False,
            dual_cls=True,
            train_cfg=_no_presence_train_cfg())
        cls_prev = torch.tensor([[5.0, -3.0, -3.0]])
        cls_curr = torch.tensor([[-3.0, 0.2, -3.0]])
        bbox_prev = torch.rand(1, 5)
        bbox_curr = torch.rand(1, 5)

        result = head._predict_by_feat_single_dual_cls(
            cls_prev, cls_curr, bbox_prev, bbox_curr, IMG_META)

        self.assertEqual(result.labels_prev.item(), 0)
        self.assertEqual(result.labels_curr.item(), 1)
        self.assertEqual(result.labels.item(), 0)

    def test_aux_layer_loss_keys(self):
        head = _build_head(num_layers=2)
        b, q = 1, 2
        losses = head.loss_by_feat(
            torch.randn(2, 1, q, 3),
            torch.randn(2, 1, q),
            torch.randn(2, 1, q),
            torch.rand(2, 1, q, 5),
            torch.rand(2, 1, q, 5),
            batch_pair_gt_instances=[_pair_gt(
                [0], [_norm_rbox(0.5, 0.5, 0.2, 0.2)],
                [_norm_rbox(0.5, 0.5, 0.2, 0.2)], [True], [True])],
            batch_img_metas=[IMG_META],
        )
        self.assertIn('d0.loss_cls', losses)
        self.assertIn('loss_cls', losses)
        self.assertNotIn('enc_loss_cls', losses)
        self.assertNotIn('dn_loss_cls', losses)

    def test_static_import_from_package(self):
        from projects.multispec_pair_rotated_rtdetr import (
            multispec_pair_rotated_rtdetr as pkg)
        self.assertTrue(hasattr(pkg, 'PairRotatedRTDETRHead'))
        self.assertTrue(hasattr(pkg, 'PairHungarianAssigner'))

    def test_config_build_minimal_forward(self):
        head = _build_head(num_layers=1, embed_dims=16)
        decoder, reg_prev, reg_curr = _build_pair_decoder(
            num_layers=1, num_queries=3, embed_dims=16)
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            torch.device('cpu'))
        mem_prev = torch.randn(1, num_value, 16)
        mem_curr = torch.randn(1, num_value, 16)
        hidden, ref_prev, ref_curr = decoder(
            mem_prev, mem_curr, spatial_shapes, level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        cls, pres_p, pres_c, bbox_p, bbox_c = head.forward(
            hidden, ref_prev, ref_curr)
        self.assertEqual(cls.shape[0], 1)
        self.assertEqual(bbox_p.shape[-1], 5)


def _spatial_meta(device: torch.device):
    spatial_shapes = torch.tensor(
        [[20, 25], [10, 13], [5, 7]], device=device, dtype=torch.long)
    level_start_index = torch.cat([
        spatial_shapes.new_zeros((1, )),
        spatial_shapes.prod(1).cumsum(0)[:-1],
    ])
    num_value = int(spatial_shapes.prod(1).sum())
    return spatial_shapes, level_start_index, num_value


def _build_pair_decoder(num_layers: int, num_queries: int, embed_dims: int):
    layer_cfg = dict(
        self_attn_cfg=dict(
            embed_dims=embed_dims, num_heads=4, dropout=0.0,
            batch_first=True),
        cross_attn_cfg=dict(
            embed_dims=embed_dims, num_heads=4, num_levels=3,
            num_points=4, dropout=0.0, batch_first=True),
        ffn_cfg=dict(
            embed_dims=embed_dims, feedforward_channels=64,
            ffn_drop=0.0, act_cfg=dict(type='GELU')),
    )
    decoder = PairRotatedRTDETRTransformerDecoder(
        num_layers=num_layers,
        num_queries=num_queries,
        return_intermediate=True,
        layer_cfg=layer_cfg,
        post_norm_cfg=None,
        angle_factor=ANGLE_FACTOR,
    )
    reg_branches_prev = torch.nn.ModuleList([
        torch.nn.Linear(embed_dims, 5) for _ in range(num_layers)
    ])
    reg_branches_curr = torch.nn.ModuleList([
        torch.nn.Linear(embed_dims, 5) for _ in range(num_layers)
    ])
    return decoder, reg_branches_prev, reg_branches_curr


if __name__ == '__main__':
    unittest.main()
