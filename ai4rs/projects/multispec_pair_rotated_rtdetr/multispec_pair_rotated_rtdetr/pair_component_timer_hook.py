# Copyright (c) AI4RS. All rights reserved.
"""Log per-component forward/backward timing during Pair RT-DETR training."""

from __future__ import annotations

import time
from typing import Dict, Optional

import torch
from mmengine.dist import get_dist_info
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmrotate.registry import HOOKS


@HOOKS.register_module()
class PairComponentTimerHook(Hook):
    """Print component timings every ``interval`` train iterations.

    Forward breakdown is produced by :class:`MultispecPairRotatedRTDETR`
    during ``loss()``. This hook adds wall-clock ``backward_opt`` and
    ``iter_wall`` for the full train step (forward + backward + optim).
    """

    priority = 'NORMAL'

    def __init__(self, interval: int = 50) -> None:
        self.interval = interval
        self._iter_start: Optional[float] = None

    @staticmethod
    def _unwrap_model(runner: Runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        return model

    def before_train_iter(self,
                          runner: Runner,
                          batch_idx: int,
                          data_batch=None) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._iter_start = time.perf_counter()

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        model = self._unwrap_model(runner)
        forward_timings: Dict[str, float] = getattr(
            model, '_last_component_timings', None)
        if not forward_timings or self._iter_start is None:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        iter_wall = time.perf_counter() - self._iter_start
        timings = dict(forward_timings)
        # assign_* entries are nested sub-timers inside head_loss. Exclude them
        # from the top-level forward sum so backward_opt is not underreported.
        forward_total = sum(
            value for key, value in timings.items()
            if not key.startswith('assign_'))
        timings['backward_opt'] = max(0.0, iter_wall - forward_total)
        timings['iter_wall'] = iter_wall

        for key, value in timings.items():
            runner.message_hub.update_scalar(f'time/{key}', value)

        if not self.every_n_train_iters(runner, self.interval):
            return

        rank, _ = get_dist_info()
        if rank != 0:
            return

        ordered = [
            'backbone_neck',
            'pre_transformer',
            'encoder',
            'encoder_to_fp32',
            'query_init',
            'decoder',
            'decoder_prev',
            'decoder_curr',
            'head_loss',
            'backward_opt',
            'iter_wall',
        ]
        parts = []
        for key in ordered:
            if key in timings:
                parts.append(f'{key}={timings[key]:.4f}s')
        for key, value in timings.items():
            if key not in ordered:
                parts.append(f'{key}={value:.4f}s')
        runner.logger.info('[ComponentTimer] ' + '  '.join(parts))
