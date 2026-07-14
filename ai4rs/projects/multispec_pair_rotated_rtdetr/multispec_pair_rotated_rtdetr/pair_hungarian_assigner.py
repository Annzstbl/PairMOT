# Copyright (c) AI4RS. All rights reserved.
"""One-to-one Hungarian assigner for pair queries vs pair GT (M4)."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Union

import logging
import torch
from mmengine.config import ConfigDict
from mmengine.logging import print_log
from mmengine.structures import InstanceData
from mmdet.models.task_modules import AssignResult
from mmdet.models.task_modules.assigners import BaseAssigner
from mmrotate.registry import TASK_UTILS
from scipy.optimize import linear_sum_assignment


@TASK_UTILS.register_module()
class PairHungarianAssigner(BaseAssigner):
    """Match each pair query to at most one complete GT pair.

    All configured match costs are summed into a single cost matrix, then
    ``linear_sum_assignment`` is executed once per image.
    """

    def __init__(
        self,
        match_costs: Union[List[Union[dict, ConfigDict]], dict, ConfigDict],
        profile_costs: bool = False,
        sanitize_nonfinite: bool = True,
    ) -> None:
        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        assert len(match_costs) > 0, 'match_costs must not be empty.'
        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]
        self.profile_costs = profile_costs
        self.sanitize_nonfinite = sanitize_nonfinite
        self._timings: Dict[str, float] = {}

    def _record_time(self, name: str, elapsed: float) -> None:
        self._timings[name] = self._timings.get(name, 0.0) + elapsed

    def pop_timings(self) -> Dict[str, float]:
        timings = dict(self._timings)
        self._timings.clear()
        return timings

    @staticmethod
    def _cost_timer_name(cost, index: int) -> str:
        side = getattr(cost, 'side', None)
        cls_name = cost.__class__.__name__
        if cls_name == 'FocalLossCost':
            return 'assign_focal'
        if cls_name == 'PairChamferCost':
            return f'assign_chamfer_{side}'
        if cls_name == 'PairGDCost':
            return f'assign_gd_{side}'
        if cls_name == 'PairPresenceBCECost':
            return f'assign_presence_{side}'
        return f'assign_cost_{index}_{cls_name}'

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        num_gts = len(gt_instances.labels)
        num_preds = pred_instances.scores.size(0)
        device = pred_instances.scores.device
        assigned_gt_inds = torch.full(
            (num_preds, ), -1, dtype=torch.long, device=device)
        assigned_labels = torch.full(
            (num_preds, ), -1, dtype=torch.long, device=device)
        if num_gts == 0 or num_preds == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        cost = None
        for index, cost_fn in enumerate(self.match_costs):
            if self.profile_costs and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            cost_i = cost_fn(pred_instances, gt_instances, img_meta, **kwargs)
            if self.sanitize_nonfinite and not torch.isfinite(cost_i).all():
                finite = cost_i[torch.isfinite(cost_i)]
                replacement = (
                    finite.max() + 1e6 if finite.numel() > 0
                    else cost_i.new_tensor(1e6))
                cost_i = torch.where(torch.isfinite(cost_i), cost_i,
                                     replacement)
                if not getattr(self, '_warned_nonfinite_cost', False):
                    print_log(
                        'PairHungarianAssigner replaced non-finite match '
                        f'cost values from {self._cost_timer_name(cost_fn, index)} '
                        'with a large finite penalty.',
                        logger='current',
                        level=logging.WARNING)
                    self._warned_nonfinite_cost = True
            cost = cost_i if cost is None else cost + cost_i
            if self.profile_costs:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                self._record_time(
                    self._cost_timer_name(cost_fn, index),
                    time.perf_counter() - start)
        if self.sanitize_nonfinite and not torch.isfinite(cost).all():
            finite = cost[torch.isfinite(cost)]
            replacement = (
                finite.max() + 1e6 if finite.numel() > 0
                else cost.new_tensor(1e6))
            cost = torch.where(torch.isfinite(cost), cost, replacement)
            if not getattr(self, '_warned_nonfinite_total_cost', False):
                print_log(
                    'PairHungarianAssigner replaced non-finite summed match '
                    'cost values with a large finite penalty.',
                    logger='current',
                    level=logging.WARNING)
                self._warned_nonfinite_total_cost = True
        if self.profile_costs and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        cost = cost.detach().cpu()
        if self.profile_costs:
            self._record_time('assign_cpu_copy', time.perf_counter() - start)
        if linear_sum_assignment is None:
            raise ImportError(
                'PairHungarianAssigner requires scipy.optimize.linear_sum_assignment')

        start = time.perf_counter()
        matched_row, matched_col = linear_sum_assignment(cost)
        if self.profile_costs:
            self._record_time(
                'assign_cpu_hungarian', time.perf_counter() - start)
        matched_row = torch.from_numpy(matched_row).to(device)
        matched_col = torch.from_numpy(matched_col).to(device)
        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row] = matched_col + 1
        assigned_labels[matched_row] = gt_instances.labels[matched_col]
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)
