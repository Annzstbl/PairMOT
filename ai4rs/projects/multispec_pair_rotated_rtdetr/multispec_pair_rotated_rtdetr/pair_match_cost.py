# Copyright (c) AI4RS. All rights reserved.
"""Pair Hungarian match costs (M4).

Reuse O2-RTDETR ``FocalLossCost`` / ``ChamferCost`` / ``GDCost`` on each frame
side; gate box costs with GT ``valid_*`` masks; add presence BCE costs.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from mmengine.structures import InstanceData
from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost
from mmrotate.registry import TASK_UTILS
from projects.rotated_dino.rotated_dino.match_cost import ChamferCost, GDCost
from torch import Tensor


def _side_fields(side: str) -> tuple[str, str, str]:
    if side == 'prev':
        return 'bboxes_prev', 'bboxes_prev', 'valid_prev'
    if side == 'curr':
        return 'bboxes_curr', 'bboxes_curr', 'valid_curr'
    raise ValueError(f'Unknown pair side {side!r}, expected "prev" or "curr".')


class PairSideBoxMatchCost(BaseMatchCost):
    """Wrap a single-frame OBB match cost for one pair side.

    Box cost is zeroed for GT columns whose ``valid_*`` is False.
    """

    def __init__(self,
                 side: str,
                 box_cost: Union[dict, ChamferCost, GDCost],
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        assert side in ('prev', 'curr')
        self.side = side
        if isinstance(box_cost, dict):
            self.box_cost = TASK_UTILS.build(box_cost)
        else:
            self.box_cost = box_cost

    @torch.no_grad()
    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        pred_key, gt_key, valid_key = _side_fields(self.side)
        num_gt = len(gt_instances.labels)
        num_pred = pred_instances.scores.size(0)
        if num_gt == 0:
            return pred_instances.scores.new_zeros((num_pred, 0))

        pred_side = InstanceData(
            bboxes=getattr(pred_instances, pred_key).float())
        valid = getattr(gt_instances, valid_key).bool()  # (num_gt,)
        gt_side = InstanceData(
            bboxes=getattr(gt_instances, gt_key)[valid].float())
        with torch.cuda.amp.autocast(enabled=False):
            valid_cost = self.box_cost(
                pred_side, gt_side, img_meta, **kwargs)
        cost = valid_cost.new_zeros((num_pred, num_gt))
        cost[:, valid] = valid_cost
        return cost * self.weight


@TASK_UTILS.register_module()
class PairChamferCost(PairSideBoxMatchCost):
    """Chamfer OBB cost for prev or curr side with ``valid_*`` gating."""

    def __init__(self,
                 side: str,
                 box_format: str = 'xywha',
                 weight: Union[float, int] = 5.0) -> None:
        super().__init__(
            side=side,
            box_cost=ChamferCost(box_format=box_format, weight=1.0),
            weight=weight)


@TASK_UTILS.register_module()
class PairGDCost(PairSideBoxMatchCost):
    """Gaussian-distance OBB cost for prev or curr side with ``valid_*`` gating."""

    def __init__(self,
                 side: str,
                 loss_type: str = 'kld',
                 fun: str = 'log1p',
                 tau: float = 1,
                 sqrt: bool = False,
                 weight: Union[float, int] = 2.0) -> None:
        super().__init__(
            side=side,
            box_cost=GDCost(
                loss_type=loss_type,
                fun=fun,
                tau=tau,
                sqrt=sqrt,
                weight=1.0),
            weight=weight)


@TASK_UTILS.register_module()
class PairPresenceBCECost(BaseMatchCost):
    """Binary presence cost between query logits and GT ``valid_*`` flags."""

    def __init__(self,
                 side: str,
                 eps: float = 1e-6,
                 weight: Union[float, int] = 1.0) -> None:
        super().__init__(weight=weight)
        assert side in ('prev', 'curr')
        self.side = side
        self.eps = eps

    @torch.no_grad()
    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        del img_meta, kwargs
        num_gt = len(gt_instances.labels)
        num_pred = pred_instances.scores.size(0)
        if num_gt == 0:
            return pred_instances.scores.new_zeros((num_pred, 0))

        if self.side == 'prev':
            pred_logit = pred_instances.presence_prev  # (num_pred,)
            gt_valid = gt_instances.valid_prev
        else:
            pred_logit = pred_instances.presence_curr  # (num_pred,)
            gt_valid = gt_instances.valid_curr

        pred_prob = pred_logit.sigmoid().unsqueeze(1)  # (num_pred, 1)
        gt_target = gt_valid.to(pred_prob.dtype).unsqueeze(0)  # (1, num_gt)
        # (num_pred, num_gt)
        bce = -(
            gt_target * (pred_prob + self.eps).log() +
            (1 - gt_target) * (1 - pred_prob + self.eps).log())
        return bce * self.weight
