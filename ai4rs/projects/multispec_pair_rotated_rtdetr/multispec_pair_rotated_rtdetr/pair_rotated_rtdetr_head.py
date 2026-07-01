# Copyright (c) AI4RS. All rights reserved.
"""Pair RT-DETR detection head with pair-frame OBB outputs."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmdet.models.utils import multi_apply
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from mmengine.structures import InstanceData
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes, qbox2rbox, rbbox_overlaps
from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETRHead
from projects.rotated_rtdetr.rotated_rtdetr.prob_iou import probiou
from projects.rotated_rtdetr.rotated_rtdetr.varifocal_loss import VarifocalLoss
from torch import Tensor

from .pair_instance_data import PairInstanceData


def _to_rbox_tensor(bboxes, angle_cfg: dict) -> Tensor:
    """Convert GT boxes to ``(N, 5)`` rbox tensor."""
    def _regularize(tensor: Tensor) -> Tensor:
        boxes = RotatedBoxes(tensor.clone())
        boxes.regularize_boxes(**angle_cfg)
        return boxes.tensor

    if isinstance(bboxes, RotatedBoxes):
        return _regularize(bboxes.tensor)
    if hasattr(bboxes, 'tensor'):
        tensor = bboxes.tensor
        if tensor.size(-1) == 8:
            tensor = qbox2rbox(tensor)
        return _regularize(tensor)
    tensor = bboxes
    if tensor.size(-1) == 8:
        tensor = qbox2rbox(tensor)
    return _regularize(tensor)


@MODELS.register_module()
class PairRotatedRTDETRHead(RotatedRTDETRHead):
    """Head for pair queries: shared cls + dual presence + dual OBB.

    Expects decoder outputs ``hidden_states`` plus per-layer
    ``references_prev`` / ``references_curr`` (sigmoid 5D OBB). Encoder
    auxiliary loss remains disabled; optional PairDN loss is supported.
    """

    def __init__(self,
                 *args,
                 loss_presence: Optional[dict] = None,
                 dn_loss_weight: float = 1.0,
                 use_presence: bool = True,
                 dual_cls: bool = False,
                 train_both_visible_only: bool = False,
                 **kwargs) -> None:
        self.use_presence = bool(use_presence)
        self.dual_cls = bool(dual_cls)
        self.train_both_visible_only = bool(train_both_visible_only)
        if loss_presence is None:
            loss_presence = dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0)
        self.loss_presence_cfg = loss_presence
        self.dn_loss_weight = dn_loss_weight
        loss_cls = kwargs.get('loss_cls')
        if isinstance(loss_cls, dict):
            self.varifocal_loss_iou_type = loss_cls.pop(
                'varifocal_loss_iou_type', 'hbox_iou')
        else:
            self.varifocal_loss_iou_type = 'hbox_iou'
        super(RotatedRTDETRHead, self).__init__(*args, **kwargs)
        self.loss_presence = MODELS.build(loss_presence) if self.use_presence else None

    def _init_layers(self) -> None:
        super()._init_layers()
        if self.dual_cls:
            self.cls_branches_curr = nn.ModuleList([
                copy.deepcopy(branch) for branch in self.cls_branches
            ])
        self.reg_branches_curr = nn.ModuleList([
            copy.deepcopy(branch) for branch in self.reg_branches
        ])
        num_layers = self.num_pred_layer
        if self.use_presence:
            pres_branch = Linear(self.embed_dims, 1)
            self.presence_prev_branches = nn.ModuleList([
                copy.deepcopy(pres_branch) for _ in range(num_layers)
            ])
            self.presence_curr_branches = nn.ModuleList([
                copy.deepcopy(pres_branch) for _ in range(num_layers)
            ])

    def _sync_reg_branches_curr_from_prev(self) -> None:
        """Mirror prev reg weights onto curr (parallel branch init)."""
        for curr_branch, prev_branch in zip(
                self.reg_branches_curr, self.reg_branches):
            curr_branch.load_state_dict(prev_branch.state_dict())

    def _sync_cls_branches_curr_from_prev(self) -> None:
        """Mirror prev cls weights onto curr when dual frame cls is enabled."""
        if not self.dual_cls:
            return
        for curr_branch, prev_branch in zip(
                self.cls_branches_curr, self.cls_branches):
            curr_branch.load_state_dict(prev_branch.state_dict())

    def init_weights(self) -> None:
        super().init_weights()
        self._sync_cls_branches_curr_from_prev()
        self._sync_reg_branches_curr_from_prev()

    def load_state_dict(self, state_dict, strict: bool = True):
        has_curr = any(
            key.startswith('reg_branches_curr.') for key in state_dict)
        has_cls_curr = any(
            key.startswith('cls_branches_curr.') for key in state_dict)
        incompatible = super().load_state_dict(state_dict, strict=False)
        if not has_cls_curr:
            self._sync_cls_branches_curr_from_prev()
        if not has_curr:
            self._sync_reg_branches_curr_from_prev()
        if strict:
            missing_keys = list(incompatible.missing_keys)
            if not has_cls_curr:
                missing_keys = [
                    key for key in missing_keys
                    if not key.startswith('cls_branches_curr.')
                ]
            if not has_curr:
                missing_keys = [
                    key for key in missing_keys
                    if not key.startswith('reg_branches_curr.')
                ]
            if missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    'Error(s) in loading state_dict for '
                    f'{self.__class__.__name__}:\n'
                    f'Missing key(s): {missing_keys}.\n'
                    f'Unexpected key(s): {incompatible.unexpected_keys}.')
        return incompatible

    def forward(
        self,
        hidden_states: List[Tensor],
        references_prev: List[Tensor],
        references_curr: List[Tensor],
    ) -> Tuple[Tensor, ...]:
        """Build per-layer pair head outputs.

        Args:
            hidden_states (list[Tensor]): Decoder hidden states per layer,
                each ``(bs, num_queries, embed_dims)``.
            references_prev (list[Tensor]): Sigmoid prev OBB per layer,
                each ``(bs, num_queries, 5)``.
            references_curr (list[Tensor]): Sigmoid curr OBB per layer,
                each ``(bs, num_queries, 5)``.

        Returns:
            tuple[Tensor]:
                - all_layers_cls_scores ``(num_layers, bs, Q, cls_out)``.
                - all_layers_presence_prev ``(num_layers, bs, Q)``.
                - all_layers_presence_curr ``(num_layers, bs, Q)``.
                - all_layers_bbox_prev ``(num_layers, bs, Q, 5)``.
                - all_layers_bbox_curr ``(num_layers, bs, Q, 5)``.
        """
        all_cls: List[Tensor] = []
        all_cls_curr: List[Tensor] = []
        all_pres_prev: List[Tensor] = []
        all_pres_curr: List[Tensor] = []
        all_bbox_prev: List[Tensor] = []
        all_bbox_curr: List[Tensor] = []

        for layer_id, hidden_state in enumerate(hidden_states):
            # hidden_state: (bs, num_queries, embed_dims)
            all_cls.append(self.cls_branches[layer_id](hidden_state))
            if self.dual_cls:
                all_cls_curr.append(
                    self.cls_branches_curr[layer_id](hidden_state))
            if self.use_presence:
                all_pres_prev.append(
                    self.presence_prev_branches[layer_id](hidden_state).squeeze(-1))
                all_pres_curr.append(
                    self.presence_curr_branches[layer_id](hidden_state).squeeze(-1))
            all_bbox_prev.append(references_prev[layer_id])
            all_bbox_curr.append(references_curr[layer_id])

        if self.dual_cls:
            if self.use_presence:
                return (
                    torch.stack(all_cls),
                    torch.stack(all_cls_curr),
                    torch.stack(all_pres_prev),
                    torch.stack(all_pres_curr),
                    torch.stack(all_bbox_prev),
                    torch.stack(all_bbox_curr),
                )
            return (
                torch.stack(all_cls),
                torch.stack(all_cls_curr),
                torch.stack(all_bbox_prev),
                torch.stack(all_bbox_curr),
            )
        if self.use_presence:
            return (
                torch.stack(all_cls),
                torch.stack(all_pres_prev),
                torch.stack(all_pres_curr),
                torch.stack(all_bbox_prev),
                torch.stack(all_bbox_curr),
            )
        return (
            torch.stack(all_cls),
            torch.stack(all_bbox_prev),
            torch.stack(all_bbox_curr),
        )

    def _filter_both_visible_gt(self, pair_gt: InstanceData) -> InstanceData:
        """Keep only GT tracks visible in both frames for the new pair task."""
        if not self.train_both_visible_only:
            return pair_gt
        valid = pair_gt.valid_prev.bool() & pair_gt.valid_curr.bool()
        if valid.all():
            return pair_gt
        filtered = InstanceData()
        for key in pair_gt.keys():
            value = getattr(pair_gt, key)
            if hasattr(value, '__getitem__') and len(value) == len(valid):
                setattr(filtered, key, value[valid])
            else:
                setattr(filtered, key, value)
        return filtered

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        *args,
        batch_pair_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        enc_cls_scores: Optional[Tensor] = None,
        enc_bbox_preds: Optional[Tensor] = None,
        dn_meta: Optional[Dict[str, int]] = None,
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> Dict[str, Tensor]:
        """Pair matching loss plus optional track-union denoising loss."""
        del enc_cls_scores, enc_bbox_preds, batch_gt_instances_ignore
        batch_pair_gt_instances = [
            self._filter_both_visible_gt(gt) for gt in batch_pair_gt_instances
        ]

        if self.dual_cls and not self.use_presence:
            if dn_meta is not None and dn_meta['num_denoising_queries'] > 0:
                raise NotImplementedError(
                    'PairDN is not implemented for dual-cls/no-presence head.')
            (all_layers_cls_curr_scores, all_layers_bbox_prev,
             all_layers_bbox_curr) = args
            return self.loss_by_feat_dual_cls(
                all_layers_cls_scores,
                all_layers_cls_curr_scores,
                all_layers_bbox_prev,
                all_layers_bbox_curr,
                batch_pair_gt_instances,
                batch_img_metas)

        if not self.use_presence:
            if dn_meta is not None and dn_meta['num_denoising_queries'] > 0:
                raise NotImplementedError(
                    'PairDN is not implemented for no-presence head.')
            all_layers_bbox_prev, all_layers_bbox_curr = args
            zero_pres_prev = all_layers_cls_scores.new_zeros(
                all_layers_cls_scores.shape[:3])
            zero_pres_curr = all_layers_cls_scores.new_zeros(
                all_layers_cls_scores.shape[:3])
            all_layers_presence_prev = zero_pres_prev
            all_layers_presence_curr = zero_pres_curr
        else:
            (all_layers_presence_prev, all_layers_presence_curr,
             all_layers_bbox_prev, all_layers_bbox_curr) = args

        if dn_meta is not None and dn_meta['num_denoising_queries'] > 0:
            num_dn = dn_meta['num_denoising_queries']
            dn_outs = (
                all_layers_cls_scores[:, :, :num_dn],
                all_layers_presence_prev[:, :, :num_dn],
                all_layers_presence_curr[:, :, :num_dn],
                all_layers_bbox_prev[:, :, :num_dn],
                all_layers_bbox_curr[:, :, :num_dn],
            )
            all_layers_cls_scores = all_layers_cls_scores[:, :, num_dn:]
            all_layers_presence_prev = all_layers_presence_prev[:, :, num_dn:]
            all_layers_presence_curr = all_layers_presence_curr[:, :, num_dn:]
            all_layers_bbox_prev = all_layers_bbox_prev[:, :, num_dn:]
            all_layers_bbox_curr = all_layers_bbox_curr[:, :, num_dn:]
        else:
            dn_outs = None

        layer_outs = multi_apply(
            self.loss_by_feat_single,
            all_layers_cls_scores,
            all_layers_presence_prev,
            all_layers_presence_curr,
            all_layers_bbox_prev,
            all_layers_bbox_curr,
            batch_pair_gt_instances=batch_pair_gt_instances,
            batch_img_metas=batch_img_metas,
        )
        (losses_cls, losses_pres_prev, losses_pres_curr, losses_bbox_prev,
         losses_bbox_curr, losses_iou_prev,
         losses_iou_curr) = layer_outs

        loss_dict = dict(
            loss_cls=losses_cls[-1],
            loss_pres_prev=losses_pres_prev[-1],
            loss_pres_curr=losses_pres_curr[-1],
            loss_bbox_prev=losses_bbox_prev[-1],
            loss_bbox_curr=losses_bbox_curr[-1],
            loss_iou_prev=losses_iou_prev[-1],
            loss_iou_curr=losses_iou_curr[-1],
        )
        num_layers = len(losses_cls) - 1
        for layer_id in range(num_layers):
            prefix = f'd{layer_id}.'
            loss_dict[f'{prefix}loss_cls'] = losses_cls[layer_id]
            loss_dict[f'{prefix}loss_pres_prev'] = losses_pres_prev[layer_id]
            loss_dict[f'{prefix}loss_pres_curr'] = losses_pres_curr[layer_id]
            loss_dict[f'{prefix}loss_bbox_prev'] = losses_bbox_prev[layer_id]
            loss_dict[f'{prefix}loss_bbox_curr'] = losses_bbox_curr[layer_id]
            loss_dict[f'{prefix}loss_iou_prev'] = losses_iou_prev[layer_id]
            loss_dict[f'{prefix}loss_iou_curr'] = losses_iou_curr[layer_id]

        if dn_outs is not None:
            dn_losses = self.loss_pair_dn(
                *dn_outs,
                batch_pair_gt_instances=batch_pair_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            (dn_cls, dn_pres_prev, dn_pres_curr, dn_bbox_prev, dn_bbox_curr,
             dn_iou_prev, dn_iou_curr) = dn_losses
            if self.dn_loss_weight != 1.0:
                dn_cls = [loss * self.dn_loss_weight for loss in dn_cls]
                dn_pres_prev = [loss * self.dn_loss_weight for loss in dn_pres_prev]
                dn_pres_curr = [loss * self.dn_loss_weight for loss in dn_pres_curr]
                dn_bbox_prev = [loss * self.dn_loss_weight for loss in dn_bbox_prev]
                dn_bbox_curr = [loss * self.dn_loss_weight for loss in dn_bbox_curr]
                dn_iou_prev = [loss * self.dn_loss_weight for loss in dn_iou_prev]
                dn_iou_curr = [loss * self.dn_loss_weight for loss in dn_iou_curr]
            loss_dict.update(
                dn_loss_cls=dn_cls[-1],
                dn_loss_pres_prev=dn_pres_prev[-1],
                dn_loss_pres_curr=dn_pres_curr[-1],
                dn_loss_bbox_prev=dn_bbox_prev[-1],
                dn_loss_bbox_curr=dn_bbox_curr[-1],
                dn_loss_iou_prev=dn_iou_prev[-1],
                dn_loss_iou_curr=dn_iou_curr[-1])
            for layer_id in range(len(dn_cls) - 1):
                prefix = f'd{layer_id}.'
                loss_dict[f'{prefix}dn_loss_cls'] = dn_cls[layer_id]
                loss_dict[f'{prefix}dn_loss_pres_prev'] = dn_pres_prev[layer_id]
                loss_dict[f'{prefix}dn_loss_pres_curr'] = dn_pres_curr[layer_id]
                loss_dict[f'{prefix}dn_loss_bbox_prev'] = dn_bbox_prev[layer_id]
                loss_dict[f'{prefix}dn_loss_bbox_curr'] = dn_bbox_curr[layer_id]
                loss_dict[f'{prefix}dn_loss_iou_prev'] = dn_iou_prev[layer_id]
                loss_dict[f'{prefix}dn_loss_iou_curr'] = dn_iou_curr[layer_id]
        return loss_dict

    def loss_by_feat_dual_cls(
        self,
        all_layers_cls_prev: Tensor,
        all_layers_cls_curr: Tensor,
        all_layers_bbox_prev: Tensor,
        all_layers_bbox_curr: Tensor,
        batch_pair_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> Dict[str, Tensor]:
        """Loss for dual per-frame cls without presence branches."""
        layer_outs = multi_apply(
            self.loss_by_feat_single_dual_cls,
            all_layers_cls_prev,
            all_layers_cls_curr,
            all_layers_bbox_prev,
            all_layers_bbox_curr,
            batch_pair_gt_instances=batch_pair_gt_instances,
            batch_img_metas=batch_img_metas,
        )
        (losses_cls_prev, losses_cls_curr, losses_bbox_prev,
         losses_bbox_curr, losses_iou_prev, losses_iou_curr) = layer_outs
        loss_dict = dict(
            loss_cls_prev=losses_cls_prev[-1],
            loss_cls_curr=losses_cls_curr[-1],
            loss_cls=losses_cls_prev[-1] + losses_cls_curr[-1],
            loss_bbox_prev=losses_bbox_prev[-1],
            loss_bbox_curr=losses_bbox_curr[-1],
            loss_iou_prev=losses_iou_prev[-1],
            loss_iou_curr=losses_iou_curr[-1],
        )
        num_layers = len(losses_cls_prev) - 1
        for layer_id in range(num_layers):
            prefix = f'd{layer_id}.'
            loss_dict[f'{prefix}loss_cls_prev'] = losses_cls_prev[layer_id]
            loss_dict[f'{prefix}loss_cls_curr'] = losses_cls_curr[layer_id]
            loss_dict[f'{prefix}loss_cls'] = (
                losses_cls_prev[layer_id] + losses_cls_curr[layer_id])
            loss_dict[f'{prefix}loss_bbox_prev'] = losses_bbox_prev[layer_id]
            loss_dict[f'{prefix}loss_bbox_curr'] = losses_bbox_curr[layer_id]
            loss_dict[f'{prefix}loss_iou_prev'] = losses_iou_prev[layer_id]
            loss_dict[f'{prefix}loss_iou_curr'] = losses_iou_curr[layer_id]
        return loss_dict

    def loss_by_feat_single_dual_cls(
        self,
        cls_prev: Tensor,
        cls_curr: Tensor,
        bbox_prev: Tensor,
        bbox_curr: Tensor,
        batch_pair_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> Tuple[Tensor, ...]:
        """Loss for one decoder layer with prev/curr cls logits."""
        num_imgs = cls_prev.size(0)
        pair_cls = 0.5 * (cls_prev + cls_curr)
        targets = self.get_targets_no_presence(
            [pair_cls[i] for i in range(num_imgs)],
            [bbox_prev[i] for i in range(num_imgs)],
            [bbox_curr[i] for i in range(num_imgs)],
            batch_pair_gt_instances,
            batch_img_metas,
        )
        (labels_list, label_weights_list, bbox_prev_targets_list,
         bbox_prev_weights_list, bbox_curr_targets_list,
         bbox_curr_weights_list, num_total_pos, num_total_neg) = targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_prev_targets = torch.cat(bbox_prev_targets_list, 0)
        bbox_prev_weights = torch.cat(bbox_prev_weights_list, 0)
        bbox_curr_targets = torch.cat(bbox_curr_targets_list, 0)
        bbox_curr_weights = torch.cat(bbox_curr_weights_list, 0)

        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        cls_prev_flat = cls_prev.reshape(-1, self.cls_out_channels)
        cls_curr_flat = cls_curr.reshape(-1, self.cls_out_channels)
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_prev_flat.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        bbox_prev_flat = bbox_prev.reshape(-1, 5)
        bbox_curr_flat = bbox_curr.reshape(-1, 5)
        loss_cls_prev = self._loss_cls(
            cls_prev_flat, labels, label_weights, bbox_prev_flat,
            bbox_curr_flat, bbox_prev_targets, bbox_curr_targets,
            bbox_prev_weights, bbox_curr_weights, batch_img_metas,
            cls_avg_factor)
        loss_cls_curr = self._loss_cls(
            cls_curr_flat, labels, label_weights, bbox_prev_flat,
            bbox_curr_flat, bbox_prev_targets, bbox_curr_targets,
            bbox_prev_weights, bbox_curr_weights, batch_img_metas,
            cls_avg_factor)

        num_total_pos_tensor = loss_cls_prev.new_tensor([num_total_pos])
        num_total_pos_val = torch.clamp(
            reduce_mean(num_total_pos_tensor), min=1).item()
        factors = self._build_rescale_factors(batch_img_metas, bbox_prev)
        loss_iou_prev = self.loss_iou(
            bbox_prev_flat * factors,
            bbox_prev_targets * factors,
            bbox_prev_weights,
            avg_factor=num_total_pos_val)
        loss_iou_curr = self.loss_iou(
            bbox_curr_flat * factors,
            bbox_curr_targets * factors,
            bbox_curr_weights,
            avg_factor=num_total_pos_val)
        loss_bbox_prev = self.loss_bbox(
            bbox_prev_flat, bbox_prev_targets, bbox_prev_weights,
            avg_factor=num_total_pos_val)
        loss_bbox_curr = self.loss_bbox(
            bbox_curr_flat, bbox_curr_targets, bbox_curr_weights,
            avg_factor=num_total_pos_val)
        return (loss_cls_prev, loss_cls_curr, loss_bbox_prev, loss_bbox_curr,
                loss_iou_prev, loss_iou_curr)

    def get_targets_no_presence(
        self,
        cls_scores_list: List[Tensor],
        bbox_prev_list: List[Tensor],
        bbox_curr_list: List[Tensor],
        batch_pair_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> tuple:
        """Compute pair targets without presence branches."""
        (labels_list, label_weights_list, bbox_prev_targets_list,
         bbox_prev_weights_list, bbox_curr_targets_list,
         bbox_curr_weights_list, pos_inds_list,
         neg_inds_list) = multi_apply(
             self._get_targets_single_no_presence,
             cls_scores_list,
             bbox_prev_list,
             bbox_curr_list,
             batch_pair_gt_instances,
             batch_img_metas,
         )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_prev_targets_list,
                bbox_prev_weights_list, bbox_curr_targets_list,
                bbox_curr_weights_list, num_total_pos, num_total_neg)

    def _get_targets_single_no_presence(
        self,
        cls_score: Tensor,
        bbox_prev: Tensor,
        bbox_curr: Tensor,
        pair_gt_instances: InstanceData,
        img_meta: dict,
    ) -> tuple:
        """Assign one image and build regression / cls targets."""
        img_h, img_w = img_meta['img_shape']
        factor = bbox_prev.new_tensor(
            [img_w, img_h, img_w, img_h, self.angle_factor]).unsqueeze(0)
        num_queries = cls_score.size(0)
        pred_instances = InstanceData(
            scores=cls_score,
            bboxes_prev=bbox_prev * factor,
            bboxes_curr=bbox_curr * factor,
        )
        gt_bboxes_prev = _to_rbox_tensor(pair_gt_instances.bboxes_prev,
                                         self.angle_cfg)
        gt_bboxes_curr = _to_rbox_tensor(pair_gt_instances.bboxes_curr,
                                         self.angle_cfg)
        gt_instances = InstanceData(
            labels=pair_gt_instances.labels,
            bboxes_prev=gt_bboxes_prev,
            bboxes_curr=gt_bboxes_curr,
            valid_prev=pair_gt_instances.valid_prev,
            valid_curr=pair_gt_instances.valid_curr,
        )
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta,
        )
        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        labels = bbox_prev.new_full((num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = bbox_prev.new_ones(num_queries)

        valid_prev = gt_instances.valid_prev
        valid_curr = gt_instances.valid_curr
        bbox_prev_targets = torch.zeros(num_queries, 5, device=bbox_prev.device)
        bbox_curr_targets = torch.zeros(num_queries, 5, device=bbox_curr.device)
        bbox_prev_weights = torch.zeros(num_queries, 5, device=bbox_prev.device)
        bbox_curr_weights = torch.zeros(num_queries, 5, device=bbox_curr.device)
        if pos_inds.numel() > 0:
            pos_gt_prev = gt_bboxes_prev[pos_assigned_gt_inds] / factor
            pos_gt_curr = gt_bboxes_curr[pos_assigned_gt_inds] / factor
            pos_valid_prev = valid_prev[pos_assigned_gt_inds]
            pos_valid_curr = valid_curr[pos_assigned_gt_inds]
            bbox_prev_targets[pos_inds] = pos_gt_prev
            bbox_curr_targets[pos_inds] = pos_gt_curr
            bbox_prev_weights[pos_inds] = pos_valid_prev.float().unsqueeze(
                -1).repeat(1, 5)
            bbox_curr_weights[pos_inds] = pos_valid_curr.float().unsqueeze(
                -1).repeat(1, 5)
        return (labels, label_weights, bbox_prev_targets, bbox_prev_weights,
                bbox_curr_targets, bbox_curr_weights, pos_inds, neg_inds)

    def _get_pair_dn_targets(self, batch_pair_gt_instances: InstanceList,
                             batch_img_metas: List[dict],
                             dn_meta: Dict[str, int], device: torch.device):
        """Build direct targets for DN slots without Hungarian matching."""
        max_targets = dn_meta['max_num_dn_targets']
        num_groups = dn_meta['num_denoising_groups']
        num_dn = dn_meta['num_denoising_queries']
        target_lists = [[] for _ in range(9)]
        num_total_pos = 0
        for pair_gt, img_meta in zip(batch_pair_gt_instances, batch_img_metas):
            labels = torch.full((num_dn,), self.num_classes, device=device,
                                dtype=torch.long)
            label_weights = torch.zeros(num_dn, device=device)
            bbox_prev_targets = torch.zeros(num_dn, 5, device=device)
            bbox_curr_targets = torch.zeros(num_dn, 5, device=device)
            bbox_prev_weights = torch.zeros(num_dn, 5, device=device)
            bbox_curr_weights = torch.zeros(num_dn, 5, device=device)
            pres_prev_targets = torch.zeros(num_dn, device=device)
            pres_curr_targets = torch.zeros(num_dn, device=device)
            pres_weights = torch.zeros(num_dn, device=device)
            num_targets = len(pair_gt.labels)
            if num_targets > 0:
                img_h, img_w = img_meta['img_shape']
                factor = bbox_prev_targets.new_tensor(
                    [img_w, img_h, img_w, img_h, self.angle_factor])
                gt_prev = _to_rbox_tensor(pair_gt.bboxes_prev,
                                          self.angle_cfg).to(device) / factor
                gt_curr = _to_rbox_tensor(pair_gt.bboxes_curr,
                                          self.angle_cfg).to(device) / factor
                valid_prev = torch.as_tensor(
                    pair_gt.valid_prev, device=device, dtype=torch.bool)
                valid_curr = torch.as_tensor(
                    pair_gt.valid_curr, device=device, dtype=torch.bool)
                for group_idx in range(2 * num_groups):
                    start = group_idx * max_targets
                    end = start + num_targets
                    labels[start:end] = pair_gt.labels.to(device)
                    label_weights[start:end] = 1
                    bbox_prev_targets[start:end] = gt_prev
                    bbox_curr_targets[start:end] = gt_curr
                    bbox_prev_weights[start:end] = valid_prev.float().unsqueeze(
                        -1).expand(-1, 5)
                    bbox_curr_weights[start:end] = valid_curr.float().unsqueeze(
                        -1).expand(-1, 5)
                    pres_prev_targets[start:end] = valid_prev.float()
                    pres_curr_targets[start:end] = valid_curr.float()
                    pres_weights[start:end] = 1
                num_total_pos += num_targets * 2 * num_groups
            for bucket, value in zip(target_lists, (
                    labels, label_weights, bbox_prev_targets,
                    bbox_prev_weights, bbox_curr_targets, bbox_curr_weights,
                    pres_prev_targets, pres_curr_targets, pres_weights)):
                bucket.append(value)
        return (*target_lists, num_total_pos)

    def loss_pair_dn(self, all_layers_cls_scores: Tensor,
                     all_layers_presence_prev: Tensor,
                     all_layers_presence_curr: Tensor,
                     all_layers_bbox_prev: Tensor,
                     all_layers_bbox_curr: Tensor,
                     batch_pair_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     dn_meta: Dict[str, int]):
        return multi_apply(
            self._loss_pair_dn_single,
            all_layers_cls_scores,
            all_layers_presence_prev,
            all_layers_presence_curr,
            all_layers_bbox_prev,
            all_layers_bbox_curr,
            batch_pair_gt_instances=batch_pair_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta)

    def _loss_pair_dn_single(self, cls_scores: Tensor, presence_prev: Tensor,
                             presence_curr: Tensor, bbox_prev: Tensor,
                             bbox_curr: Tensor,
                             batch_pair_gt_instances: InstanceList,
                             batch_img_metas: List[dict],
                             dn_meta: Dict[str, int]):
        (labels_list, label_weights_list, bbox_prev_targets_list,
         bbox_prev_weights_list, bbox_curr_targets_list,
         bbox_curr_weights_list, pres_prev_targets_list,
         pres_curr_targets_list, pres_weights_list,
         num_total_pos) = self._get_pair_dn_targets(
             batch_pair_gt_instances, batch_img_metas, dn_meta,
             cls_scores.device)
        labels = torch.cat(labels_list)
        label_weights = torch.cat(label_weights_list)
        bbox_prev_targets = torch.cat(bbox_prev_targets_list)
        bbox_prev_weights = torch.cat(bbox_prev_weights_list)
        bbox_curr_targets = torch.cat(bbox_curr_targets_list)
        bbox_curr_weights = torch.cat(bbox_curr_weights_list)
        pres_prev_targets = torch.cat(pres_prev_targets_list)
        pres_curr_targets = torch.cat(pres_curr_targets_list)
        pres_weights = torch.cat(pres_weights_list)
        cls_flat = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = max(float(reduce_mean(
            cls_flat.new_tensor([num_total_pos])).item()), 1.0)
        loss_cls = self._loss_cls(
            cls_flat, labels, label_weights, bbox_prev.reshape(-1, 5),
            bbox_curr.reshape(-1, 5), bbox_prev_targets, bbox_curr_targets,
            bbox_prev_weights, bbox_curr_weights, batch_img_metas,
            cls_avg_factor)
        num_pos = max(float(reduce_mean(loss_cls.new_tensor(
            [num_total_pos])).item()), 1.0)
        loss_pres_prev = self.loss_presence(
            presence_prev.reshape(-1), pres_prev_targets, pres_weights,
            avg_factor=num_pos)
        loss_pres_curr = self.loss_presence(
            presence_curr.reshape(-1), pres_curr_targets, pres_weights,
            avg_factor=num_pos)
        factors = self._build_rescale_factors(batch_img_metas, bbox_prev)
        bbox_prev_flat = bbox_prev.reshape(-1, 5)
        bbox_curr_flat = bbox_curr.reshape(-1, 5)
        loss_iou_prev = self.loss_iou(
            bbox_prev_flat * factors, bbox_prev_targets * factors,
            bbox_prev_weights, avg_factor=num_pos)
        loss_iou_curr = self.loss_iou(
            bbox_curr_flat * factors, bbox_curr_targets * factors,
            bbox_curr_weights, avg_factor=num_pos)
        loss_bbox_prev = self.loss_bbox(
            bbox_prev_flat, bbox_prev_targets, bbox_prev_weights,
            avg_factor=num_pos)
        loss_bbox_curr = self.loss_bbox(
            bbox_curr_flat, bbox_curr_targets, bbox_curr_weights,
            avg_factor=num_pos)
        return (loss_cls, loss_pres_prev, loss_pres_curr, loss_bbox_prev,
                loss_bbox_curr, loss_iou_prev, loss_iou_curr)

    def loss_by_feat_single(
        self,
        cls_scores: Tensor,
        presence_prev: Tensor,
        presence_curr: Tensor,
        bbox_prev: Tensor,
        bbox_curr: Tensor,
        batch_pair_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> Tuple[Tensor, ...]:
        """Loss for one decoder layer."""
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        presence_prev_list = [presence_prev[i] for i in range(num_imgs)]
        presence_curr_list = [presence_curr[i] for i in range(num_imgs)]
        bbox_prev_list = [bbox_prev[i] for i in range(num_imgs)]
        bbox_curr_list = [bbox_curr[i] for i in range(num_imgs)]

        targets = self.get_targets(
            cls_scores_list,
            presence_prev_list,
            presence_curr_list,
            bbox_prev_list,
            bbox_curr_list,
            batch_pair_gt_instances,
            batch_img_metas,
        )
        (labels_list, label_weights_list, bbox_prev_targets_list,
         bbox_prev_weights_list, bbox_curr_targets_list,
         bbox_curr_weights_list, pres_prev_targets_list,
         pres_curr_targets_list, pres_weights_list, num_total_pos,
         num_total_neg) = targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_prev_targets = torch.cat(bbox_prev_targets_list, 0)
        bbox_prev_weights = torch.cat(bbox_prev_weights_list, 0)
        bbox_curr_targets = torch.cat(bbox_curr_targets_list, 0)
        bbox_curr_weights = torch.cat(bbox_curr_weights_list, 0)
        pres_prev_targets = torch.cat(pres_prev_targets_list, 0)
        pres_curr_targets = torch.cat(pres_curr_targets_list, 0)
        pres_weights = torch.cat(pres_weights_list, 0)

        cls_scores_flat = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores_flat.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self._loss_cls(
            cls_scores_flat,
            labels,
            label_weights,
            bbox_prev.reshape(-1, 5),
            bbox_curr.reshape(-1, 5),
            bbox_prev_targets,
            bbox_curr_targets,
            bbox_prev_weights,
            bbox_curr_weights,
            batch_img_metas,
            cls_avg_factor,
        )

        num_total_pos_tensor = loss_cls.new_tensor([num_total_pos])
        num_total_pos_val = torch.clamp(
            reduce_mean(num_total_pos_tensor), min=1).item()

        pres_prev_flat = presence_prev.reshape(-1)
        pres_curr_flat = presence_curr.reshape(-1)
        loss_pres_prev = self.loss_presence(
            pres_prev_flat,
            pres_prev_targets,
            pres_weights,
            avg_factor=num_total_pos_val + num_total_neg)
        loss_pres_curr = self.loss_presence(
            pres_curr_flat,
            pres_curr_targets,
            pres_weights,
            avg_factor=num_total_pos_val + num_total_neg)

        factors = self._build_rescale_factors(batch_img_metas, bbox_prev)

        bbox_prev_flat = bbox_prev.reshape(-1, 5)
        bbox_curr_flat = bbox_curr.reshape(-1, 5)
        bboxes_prev = bbox_prev_flat * factors
        bboxes_curr = bbox_curr_flat * factors
        bboxes_prev_gt = bbox_prev_targets * factors
        bboxes_curr_gt = bbox_curr_targets * factors

        loss_iou_prev = self.loss_iou(
            bboxes_prev,
            bboxes_prev_gt,
            bbox_prev_weights,
            avg_factor=num_total_pos_val)
        loss_iou_curr = self.loss_iou(
            bboxes_curr,
            bboxes_curr_gt,
            bbox_curr_weights,
            avg_factor=num_total_pos_val)
        loss_bbox_prev = self.loss_bbox(
            bbox_prev_flat,
            bbox_prev_targets,
            bbox_prev_weights,
            avg_factor=num_total_pos_val)
        loss_bbox_curr = self.loss_bbox(
            bbox_curr_flat,
            bbox_curr_targets,
            bbox_curr_weights,
            avg_factor=num_total_pos_val)

        return (loss_cls, loss_pres_prev, loss_pres_curr, loss_bbox_prev,
                loss_bbox_curr, loss_iou_prev, loss_iou_curr)

    def _loss_cls(
        self,
        cls_scores: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_prev: Tensor,
        bbox_curr: Tensor,
        bbox_prev_targets: Tensor,
        bbox_curr_targets: Tensor,
        bbox_prev_weights: Tensor,
        bbox_curr_weights: Tensor,
        batch_img_metas: List[dict],
        cls_avg_factor: float,
    ) -> Tensor:
        if isinstance(self.loss_cls, VarifocalLoss):
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0)
                        & (labels < bg_class_ind)).nonzero().squeeze(1)
            cls_iou_targets = label_weights.new_zeros(cls_scores.shape)
            if pos_inds.numel() > 0:
                pos_labels = labels[pos_inds]
                iou_targets = self._pair_iou_targets(
                    bbox_prev[pos_inds],
                    bbox_curr[pos_inds],
                    bbox_prev_targets[pos_inds],
                    bbox_curr_targets[pos_inds],
                    bbox_prev_weights[pos_inds],
                    bbox_curr_weights[pos_inds],
                    batch_img_metas,
                    pos_inds,
                    bbox_prev.size(0),
                )
                cls_iou_targets[pos_inds, pos_labels] = iou_targets
            return self.loss_cls(
                cls_scores, cls_iou_targets, avg_factor=cls_avg_factor)
        return self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

    def _pair_hbox_iou_targets(
        self,
        bbox_prev: Tensor,
        bbox_curr: Tensor,
        bbox_prev_targets: Tensor,
        bbox_curr_targets: Tensor,
        bbox_prev_weights: Tensor,
        bbox_curr_weights: Tensor,
    ) -> Tensor:
        """IoU targets for Varifocal cls on visible sides only."""
        valid_prev = bbox_prev_weights[:, 0] > 0
        valid_curr = bbox_curr_weights[:, 0] > 0
        iou = bbox_prev.new_zeros(bbox_prev.size(0))
        side_count = bbox_prev.new_zeros(bbox_prev.size(0))

        if valid_prev.any():
            pred_prev_xyxy = bbox_cxcywh_to_xyxy(
                bbox_prev[valid_prev, :4])
            tgt_prev_xyxy = bbox_cxcywh_to_xyxy(
                bbox_prev_targets[valid_prev, :4])
            iou[valid_prev] += bbox_overlaps(
                pred_prev_xyxy.detach(), tgt_prev_xyxy, is_aligned=True)
            side_count[valid_prev] += 1

        if valid_curr.any():
            pred_curr_xyxy = bbox_cxcywh_to_xyxy(
                bbox_curr[valid_curr, :4])
            tgt_curr_xyxy = bbox_cxcywh_to_xyxy(
                bbox_curr_targets[valid_curr, :4])
            iou[valid_curr] += bbox_overlaps(
                pred_curr_xyxy.detach(), tgt_curr_xyxy, is_aligned=True)
            side_count[valid_curr] += 1

        valid_any = side_count > 0
        iou[valid_any] = iou[valid_any] / side_count[valid_any]
        return iou

    def _pair_iou_targets(
        self,
        bbox_prev: Tensor,
        bbox_curr: Tensor,
        bbox_prev_targets: Tensor,
        bbox_curr_targets: Tensor,
        bbox_prev_weights: Tensor,
        bbox_curr_weights: Tensor,
        batch_img_metas: List[dict],
        flat_pos_inds: Tensor,
        flat_num_queries: int,
    ) -> Tensor:
        """Quality targets for pair Varifocal cls on visible sides."""
        if self.varifocal_loss_iou_type == 'hbox_iou':
            return self._pair_hbox_iou_targets(
                bbox_prev, bbox_curr, bbox_prev_targets, bbox_curr_targets,
                bbox_prev_weights, bbox_curr_weights)

        if self.varifocal_loss_iou_type not in ('rbox_iou', 'prob_iou'):
            raise NotImplementedError(
                f'Unsupported pair Varifocal IoU target '
                f'{self.varifocal_loss_iou_type!r}')

        num_imgs = len(batch_img_metas)
        num_queries = flat_num_queries // max(num_imgs, 1)
        factors = []
        for img_meta in batch_img_metas:
            img_h, img_w = img_meta['img_shape']
            factor = bbox_prev.new_tensor(
                [img_w, img_h, img_w, img_h,
                 self.angle_factor]).unsqueeze(0)
            factors.append(factor.repeat(num_queries, 1))
        pos_factors = torch.cat(factors, 0)[flat_pos_inds]

        pred_prev = bbox_prev * pos_factors
        pred_curr = bbox_curr * pos_factors
        target_prev = bbox_prev_targets * pos_factors
        target_curr = bbox_curr_targets * pos_factors

        valid_prev = bbox_prev_weights[:, 0] > 0
        valid_curr = bbox_curr_weights[:, 0] > 0
        iou = bbox_prev.new_zeros(bbox_prev.size(0))
        side_count = bbox_prev.new_zeros(bbox_prev.size(0))

        if self.varifocal_loss_iou_type == 'rbox_iou':
            overlap_fn = rbbox_overlaps
        else:
            overlap_fn = None

        if valid_prev.any():
            if overlap_fn is None:
                iou[valid_prev] += probiou(
                    pred_prev[valid_prev].detach(),
                    target_prev[valid_prev])[:, 0]
            else:
                iou[valid_prev] += overlap_fn(
                    pred_prev[valid_prev].detach(),
                    target_prev[valid_prev],
                    is_aligned=True)
            side_count[valid_prev] += 1

        if valid_curr.any():
            if overlap_fn is None:
                iou[valid_curr] += probiou(
                    pred_curr[valid_curr].detach(),
                    target_curr[valid_curr])[:, 0]
            else:
                iou[valid_curr] += overlap_fn(
                    pred_curr[valid_curr].detach(),
                    target_curr[valid_curr],
                    is_aligned=True)
            side_count[valid_curr] += 1

        valid_any = side_count > 0
        iou[valid_any] = iou[valid_any] / side_count[valid_any]
        return iou.clamp_(min=0, max=1)

    def _build_rescale_factors(self, batch_img_metas: List[dict],
                               bbox_prev: Tensor) -> Tensor:
        factors = []
        for img_meta, _ in zip(batch_img_metas, bbox_prev):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_prev.new_tensor(
                [img_w, img_h, img_w, img_h,
                 self.angle_factor]).unsqueeze(0)
            factors.append(factor)
        return torch.cat(factors, 0).repeat_interleave(
            bbox_prev.size(1), dim=0)

    def get_targets(
        self,
        cls_scores_list: List[Tensor],
        presence_prev_list: List[Tensor],
        presence_curr_list: List[Tensor],
        bbox_prev_list: List[Tensor],
        bbox_curr_list: List[Tensor],
        batch_pair_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> tuple:
        """Compute pair targets for one decoder layer."""
        (labels_list, label_weights_list, bbox_prev_targets_list,
         bbox_prev_weights_list, bbox_curr_targets_list,
         bbox_curr_weights_list, pres_prev_targets_list,
         pres_curr_targets_list, pres_weights_list, pos_inds_list,
         neg_inds_list) = multi_apply(
             self._get_targets_single,
             cls_scores_list,
             presence_prev_list,
             presence_curr_list,
             bbox_prev_list,
             bbox_curr_list,
             batch_pair_gt_instances,
             batch_img_metas,
         )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_prev_targets_list,
                bbox_prev_weights_list, bbox_curr_targets_list,
                bbox_curr_weights_list, pres_prev_targets_list,
                pres_curr_targets_list, pres_weights_list, num_total_pos,
                num_total_neg)

    def _get_targets_single(
        self,
        cls_score: Tensor,
        presence_prev: Tensor,
        presence_curr: Tensor,
        bbox_prev: Tensor,
        bbox_curr: Tensor,
        pair_gt_instances: InstanceData,
        img_meta: dict,
    ) -> tuple:
        """Assign one image and build regression / cls / presence targets."""
        img_h, img_w = img_meta['img_shape']
        factor = bbox_prev.new_tensor(
            [img_w, img_h, img_w, img_h, self.angle_factor]).unsqueeze(0)
        num_queries = cls_score.size(0)

        pred_instances = InstanceData(
            scores=cls_score,
            bboxes_prev=bbox_prev * factor,
            bboxes_curr=bbox_curr * factor,
            presence_prev=presence_prev,
            presence_curr=presence_curr,
        )

        gt_bboxes_prev = _to_rbox_tensor(pair_gt_instances.bboxes_prev,
                                         self.angle_cfg)
        gt_bboxes_curr = _to_rbox_tensor(pair_gt_instances.bboxes_curr,
                                         self.angle_cfg)
        gt_instances = InstanceData(
            labels=pair_gt_instances.labels,
            bboxes_prev=gt_bboxes_prev,
            bboxes_curr=gt_bboxes_curr,
            valid_prev=pair_gt_instances.valid_prev,
            valid_curr=pair_gt_instances.valid_curr,
        )

        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta,
        )

        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        labels = bbox_prev.new_full((num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = bbox_prev.new_ones(num_queries)

        valid_prev = gt_instances.valid_prev
        valid_curr = gt_instances.valid_curr

        pres_prev_targets = bbox_prev.new_zeros(num_queries)
        pres_curr_targets = bbox_prev.new_zeros(num_queries)
        pres_prev_targets[pos_inds] = valid_prev[pos_assigned_gt_inds].float()
        pres_curr_targets[pos_inds] = valid_curr[pos_assigned_gt_inds].float()
        pres_weights = bbox_prev.new_ones(num_queries)

        bbox_prev_targets = torch.zeros(num_queries, 5, device=bbox_prev.device)
        bbox_curr_targets = torch.zeros(num_queries, 5, device=bbox_curr.device)
        bbox_prev_weights = torch.zeros(num_queries, 5, device=bbox_prev.device)
        bbox_curr_weights = torch.zeros(num_queries, 5, device=bbox_curr.device)

        if pos_inds.numel() > 0:
            pos_gt_prev = gt_bboxes_prev[pos_assigned_gt_inds] / factor
            pos_gt_curr = gt_bboxes_curr[pos_assigned_gt_inds] / factor
            pos_valid_prev = valid_prev[pos_assigned_gt_inds]
            pos_valid_curr = valid_curr[pos_assigned_gt_inds]

            bbox_prev_targets[pos_inds] = pos_gt_prev
            bbox_curr_targets[pos_inds] = pos_gt_curr
            bbox_prev_weights[pos_inds] = pos_valid_prev.float().unsqueeze(
                -1).repeat(1, 5)
            bbox_curr_weights[pos_inds] = pos_valid_curr.float().unsqueeze(
                -1).repeat(1, 5)

        return (labels, label_weights, bbox_prev_targets, bbox_prev_weights,
                bbox_curr_targets, bbox_curr_weights, pres_prev_targets,
                pres_curr_targets, pres_weights, pos_inds, neg_inds)

    def predict_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        *args,
        batch_img_metas: List[dict],
        rescale: bool = False,
    ) -> InstanceList:
        """Transform decoder outputs into ``PairInstanceData`` (no NMS)."""
        if self.dual_cls and not self.use_presence:
            all_layers_cls_curr_scores, all_layers_bbox_prev, all_layers_bbox_curr = args
            return self.predict_by_feat_dual_cls(
                all_layers_cls_scores,
                all_layers_cls_curr_scores,
                all_layers_bbox_prev,
                all_layers_bbox_curr,
                batch_img_metas,
                rescale=rescale)

        if not self.use_presence:
            all_layers_bbox_prev, all_layers_bbox_curr = args
            all_layers_presence_prev = all_layers_cls_scores.new_ones(
                all_layers_cls_scores.shape[:3]) * 20
            all_layers_presence_curr = all_layers_cls_scores.new_ones(
                all_layers_cls_scores.shape[:3]) * 20
        else:
            (all_layers_presence_prev, all_layers_presence_curr,
             all_layers_bbox_prev, all_layers_bbox_curr) = args
        cls_scores = all_layers_cls_scores[-1]
        presence_prev = all_layers_presence_prev[-1]
        presence_curr = all_layers_presence_curr[-1]
        bbox_prev = all_layers_bbox_prev[-1]
        bbox_curr = all_layers_bbox_curr[-1]

        result_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            result_list.append(
                self._predict_by_feat_single(
                    cls_scores[img_id],
                    presence_prev[img_id],
                    presence_curr[img_id],
                    bbox_prev[img_id],
                    bbox_curr[img_id],
                    img_meta,
                    rescale=rescale))
        return result_list

    def _predict_by_feat_single(
        self,
        cls_score: Tensor,
        presence_prev: Tensor,
        presence_curr: Tensor,
        bbox_prev: Tensor,
        bbox_curr: Tensor,
        img_meta: dict,
        rescale: bool = False,
    ) -> PairInstanceData:
        """Post-process one image into ``PairInstanceData`` without NMS."""
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        max_per_img = min(max_per_img, cls_score.numel())
        img_shape = img_meta['img_shape']

        if self.loss_cls.use_sigmoid:
            # A pair query represents one object pair.  Flattening Q*C can
            # emit the same query multiple times with different labels, which
            # breaks the one-query/one-pair matching contract.
            scores, det_labels = cls_score.sigmoid().max(-1)
            max_per_img = min(max_per_img, scores.numel())
            scores, bbox_index = scores.topk(max_per_img)
            det_labels = det_labels[bbox_index]
        else:
            scores, det_labels = cls_score.softmax(dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            det_labels = det_labels[bbox_index]

        det_bboxes_prev = bbox_prev[bbox_index].clone()
        det_bboxes_curr = bbox_curr[bbox_index].clone()
        det_bboxes_prev[:, 0:4:2] *= img_shape[1]
        det_bboxes_prev[:, 1:4:2] *= img_shape[0]
        det_bboxes_prev[:, 4] *= self.angle_factor
        det_bboxes_curr[:, 0:4:2] *= img_shape[1]
        det_bboxes_curr[:, 1:4:2] *= img_shape[0]
        det_bboxes_curr[:, 4] *= self.angle_factor

        det_bboxes_prev[:, 0:4:2].clamp_(min=0, max=img_shape[1])
        det_bboxes_prev[:, 1:4:2].clamp_(min=0, max=img_shape[0])
        det_bboxes_curr[:, 0:4:2].clamp_(min=0, max=img_shape[1])
        det_bboxes_curr[:, 1:4:2].clamp_(min=0, max=img_shape[0])

        if rescale:
            scale_factor = np.array(img_meta['scale_factor']).repeat(2)
            if scale_factor.shape[0] == 4:
                scale_factor = np.append(scale_factor, 1)
            scale = det_bboxes_prev.new_tensor(scale_factor)
            det_bboxes_prev = det_bboxes_prev / scale
            det_bboxes_curr = det_bboxes_curr / scale

        results = PairInstanceData()
        results.scores = scores
        results.labels = det_labels
        results.bboxes_prev = det_bboxes_prev
        results.bboxes_curr = det_bboxes_curr
        results.presence_prev = presence_prev[bbox_index].sigmoid()
        results.presence_curr = presence_curr[bbox_index].sigmoid()
        return results

    def predict_by_feat_dual_cls(
        self,
        all_layers_cls_prev: Tensor,
        all_layers_cls_curr: Tensor,
        all_layers_bbox_prev: Tensor,
        all_layers_bbox_curr: Tensor,
        batch_img_metas: List[dict],
        rescale: bool = False,
    ) -> InstanceList:
        """Post-process dual per-frame cls outputs without presence."""
        cls_prev = all_layers_cls_prev[-1]
        cls_curr = all_layers_cls_curr[-1]
        bbox_prev = all_layers_bbox_prev[-1]
        bbox_curr = all_layers_bbox_curr[-1]
        result_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            result_list.append(
                self._predict_by_feat_single_dual_cls(
                    cls_prev[img_id],
                    cls_curr[img_id],
                    bbox_prev[img_id],
                    bbox_curr[img_id],
                    img_meta,
                    rescale=rescale))
        return result_list

    def _predict_by_feat_single_dual_cls(
        self,
        cls_prev: Tensor,
        cls_curr: Tensor,
        bbox_prev: Tensor,
        bbox_curr: Tensor,
        img_meta: dict,
        rescale: bool = False,
    ) -> PairInstanceData:
        """Post-process one image for dual cls / no presence."""
        if self.loss_cls.use_sigmoid:
            scores_prev, labels_prev = cls_prev.sigmoid().max(-1)
            scores_curr, labels_curr = cls_curr.sigmoid().max(-1)
        else:
            scores_prev, labels_prev = cls_prev.softmax(dim=-1)[..., :-1].max(-1)
            scores_curr, labels_curr = cls_curr.softmax(dim=-1)[..., :-1].max(-1)
        pair_scores = torch.sqrt(
            scores_prev.clamp(min=1e-6) * scores_curr.clamp(min=1e-6))
        max_per_img = self.test_cfg.get('max_per_img', len(pair_scores))
        max_per_img = min(max_per_img, pair_scores.numel())
        scores, bbox_index = pair_scores.topk(max_per_img)
        labels = labels_curr[bbox_index]
        img_shape = img_meta['img_shape']

        det_bboxes_prev = bbox_prev[bbox_index].clone()
        det_bboxes_curr = bbox_curr[bbox_index].clone()
        det_bboxes_prev[:, 0:4:2] *= img_shape[1]
        det_bboxes_prev[:, 1:4:2] *= img_shape[0]
        det_bboxes_prev[:, 4] *= self.angle_factor
        det_bboxes_curr[:, 0:4:2] *= img_shape[1]
        det_bboxes_curr[:, 1:4:2] *= img_shape[0]
        det_bboxes_curr[:, 4] *= self.angle_factor
        det_bboxes_prev[:, 0:4:2].clamp_(min=0, max=img_shape[1])
        det_bboxes_prev[:, 1:4:2].clamp_(min=0, max=img_shape[0])
        det_bboxes_curr[:, 0:4:2].clamp_(min=0, max=img_shape[1])
        det_bboxes_curr[:, 1:4:2].clamp_(min=0, max=img_shape[0])
        if rescale:
            scale_factor = np.array(img_meta['scale_factor']).repeat(2)
            if scale_factor.shape[0] == 4:
                scale_factor = np.append(scale_factor, 1)
            scale = det_bboxes_prev.new_tensor(scale_factor)
            det_bboxes_prev = det_bboxes_prev / scale
            det_bboxes_curr = det_bboxes_curr / scale

        results = PairInstanceData()
        results.scores = scores
        results.labels = labels
        results.bboxes_prev = det_bboxes_prev
        results.bboxes_curr = det_bboxes_curr
        results.scores_prev = scores_prev[bbox_index]
        results.scores_curr = scores_curr[bbox_index]
        results.labels_prev = labels_prev[bbox_index]
        results.labels_curr = labels_curr[bbox_index]
        return results

    @staticmethod
    def _hidden_list(hidden_states: Tensor) -> List[Tensor]:
        if isinstance(hidden_states, Tensor):
            return [hidden_states[i] for i in range(hidden_states.shape[0])]
        return hidden_states

    def loss(
        self,
        hidden_states: Tensor,
        references_prev: List[Tensor],
        references_curr: List[Tensor],
        enc_outputs_class: Optional[Tensor] = None,
        enc_outputs_coord: Optional[Tensor] = None,
        batch_data_samples: Optional[SampleList] = None,
        dn_meta: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Compute pair losses from decoder outputs."""
        del kwargs
        batch_pair_gt_instances = [
            data_sample.pair_gt_instances for data_sample in batch_data_samples
        ]
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        hidden_list = self._hidden_list(hidden_states)
        outs = self.forward(hidden_list, references_prev, references_curr)
        return self.loss_by_feat(
            *outs,
            batch_pair_gt_instances=batch_pair_gt_instances,
            batch_img_metas=batch_img_metas,
            enc_cls_scores=enc_outputs_class,
            enc_bbox_preds=enc_outputs_coord,
            dn_meta=dn_meta,
        )

    def predict(
        self,
        hidden_states: Tensor,
        references_prev: List[Tensor],
        references_curr: List[Tensor],
        batch_data_samples: SampleList,
        rescale: bool = True,
        **kwargs,
    ) -> InstanceList:
        """Run pair post-processing without NMS."""
        del kwargs
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        hidden_list = self._hidden_list(hidden_states)
        outs = self.forward(hidden_list, references_prev, references_curr)
        return self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
