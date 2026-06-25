# Copyright (c) AI4RS. All rights reserved.
"""Pair RT-DETR detection head with shared cls and dual OBB / presence (M4)."""

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
from mmrotate.structures.bbox import RotatedBoxes, qbox2rbox
from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETRHead
from projects.rotated_rtdetr.rotated_rtdetr.varifocal_loss import VarifocalLoss
from torch import Tensor

from .pair_instance_data import PairInstanceData


def _to_rbox_tensor(bboxes, angle_cfg: dict) -> Tensor:
    """Convert GT boxes to ``(N, 5)`` rbox tensor."""
    if isinstance(bboxes, RotatedBoxes):
        bboxes = bboxes.clone()
        bboxes.regularize_boxes(**angle_cfg)
        return bboxes.tensor
    if hasattr(bboxes, 'tensor'):
        tensor = bboxes.tensor
        if tensor.size(-1) == 8:
            return qbox2rbox(tensor)
        return tensor
    tensor = bboxes
    if tensor.size(-1) == 8:
        return qbox2rbox(tensor)
    return tensor


@MODELS.register_module()
class PairRotatedRTDETRHead(RotatedRTDETRHead):
    """Head for pair queries: shared cls + dual presence + dual OBB.

    Expects decoder outputs ``hidden_states`` plus per-layer
    ``references_prev`` / ``references_curr`` (sigmoid 5D OBB). Encoder
    auxiliary loss and denoising loss are disabled in ``loss_by_feat``.
    """

    def __init__(self,
                 *args,
                 loss_presence: Optional[dict] = None,
                 **kwargs) -> None:
        if loss_presence is None:
            loss_presence = dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0)
        self.loss_presence_cfg = loss_presence
        loss_cls = kwargs.get('loss_cls')
        if isinstance(loss_cls, dict):
            self.varifocal_loss_iou_type = loss_cls.pop(
                'varifocal_loss_iou_type', 'hbox_iou')
        else:
            self.varifocal_loss_iou_type = 'hbox_iou'
        super(RotatedRTDETRHead, self).__init__(*args, **kwargs)
        self.loss_presence = MODELS.build(loss_presence)

    def _init_layers(self) -> None:
        super()._init_layers()
        self.reg_branches_curr = nn.ModuleList([
            copy.deepcopy(branch) for branch in self.reg_branches
        ])
        pres_branch = Linear(self.embed_dims, 1)
        num_layers = self.num_pred_layer
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

    def init_weights(self) -> None:
        super().init_weights()
        self._sync_reg_branches_curr_from_prev()

    def load_state_dict(self, state_dict, strict: bool = True):
        has_curr = any(
            key.startswith('reg_branches_curr.') for key in state_dict)
        incompatible = super().load_state_dict(state_dict, strict=False)
        if not has_curr:
            self._sync_reg_branches_curr_from_prev()
        if strict:
            missing_keys = list(incompatible.missing_keys)
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
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        all_pres_prev: List[Tensor] = []
        all_pres_curr: List[Tensor] = []
        all_bbox_prev: List[Tensor] = []
        all_bbox_curr: List[Tensor] = []

        for layer_id, hidden_state in enumerate(hidden_states):
            # hidden_state: (bs, num_queries, embed_dims)
            all_cls.append(self.cls_branches[layer_id](hidden_state))
            all_pres_prev.append(
                self.presence_prev_branches[layer_id](hidden_state).squeeze(-1))
            all_pres_curr.append(
                self.presence_curr_branches[layer_id](hidden_state).squeeze(-1))
            all_bbox_prev.append(references_prev[layer_id])
            all_bbox_curr.append(references_curr[layer_id])

        return (
            torch.stack(all_cls),
            torch.stack(all_pres_prev),
            torch.stack(all_pres_curr),
            torch.stack(all_bbox_prev),
            torch.stack(all_bbox_curr),
        )

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_presence_prev: Tensor,
        all_layers_presence_curr: Tensor,
        all_layers_bbox_prev: Tensor,
        all_layers_bbox_curr: Tensor,
        batch_pair_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        enc_cls_scores: Optional[Tensor] = None,
        enc_bbox_preds: Optional[Tensor] = None,
        dn_meta: Optional[Dict[str, int]] = None,
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> Dict[str, Tensor]:
        """Pair matching loss over all decoder layers (no enc / DN)."""
        del enc_cls_scores, enc_bbox_preds, dn_meta, batch_gt_instances_ignore

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
        return loss_dict

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
                iou_targets = self._pair_hbox_iou_targets(
                    bbox_prev[pos_inds],
                    bbox_curr[pos_inds],
                    bbox_prev_targets[pos_inds],
                    bbox_curr_targets[pos_inds],
                    bbox_prev_weights[pos_inds],
                    bbox_curr_weights[pos_inds],
                )
                cls_iou_targets[pos_inds, pos_labels] = iou_targets
            return self.loss_cls(
                cls_scores, cls_iou_targets, avg_factor=cls_avg_factor)
        return self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

    @staticmethod
    def _pair_hbox_iou_targets(
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
        all_layers_presence_prev: Tensor,
        all_layers_presence_curr: Tensor,
        all_layers_bbox_prev: Tensor,
        all_layers_bbox_curr: Tensor,
        batch_img_metas: List[dict],
        rescale: bool = False,
    ) -> InstanceList:
        """Transform decoder outputs into ``PairInstanceData`` (no NMS)."""
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
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
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
        (all_cls, all_pres_prev, all_pres_curr, all_bbox_prev,
         all_bbox_curr) = self.forward(
             hidden_list, references_prev, references_curr)
        return self.loss_by_feat(
            all_cls,
            all_pres_prev,
            all_pres_curr,
            all_bbox_prev,
            all_bbox_curr,
            batch_pair_gt_instances,
            batch_img_metas,
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
        return self.predict_by_feat(*outs, batch_img_metas, rescale=rescale)
