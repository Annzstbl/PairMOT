# Copyright (c) AI4RS. All rights reserved.
"""Unit tests for PairRotatedRTDETRHead / PairHungarianAssigner (M4)."""

import copy
import os.path as osp
import sys
import unittest

import torch
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


def _build_head(num_layers: int = 2,
                num_classes: int = 3,
                embed_dims: int = 32,
                device: torch.device = torch.device('cpu')) -> PairRotatedRTDETRHead:
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
        train_cfg=_default_train_cfg(),
        test_cfg=dict(max_per_img=10),
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


class TestPairRotatedRTDETRHeadForward(unittest.TestCase):

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
            cls, pres_p, pres_c, bbox_p, bbox_c, [IMG_META])
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], PairInstanceData)
        self.assertEqual(results[0].bboxes_prev.shape[-1], 5)
        self.assertEqual(results[0].bboxes_curr.shape[-1], 5)

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
