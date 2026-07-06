# Copyright (c) AI4RS. All rights reserved.
"""Early stopping based on asynchronous PairMOT TrackEval metrics."""

from __future__ import annotations

import json
import math
import os
import os.path as osp
from typing import Dict, Optional

from mmengine.dist import get_dist_info
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmrotate.registry import HOOKS


@HOOKS.register_module()
class PairTrackEarlyStoppingHook(Hook):
    """Stop when both cls HOTA and det HOTA stop improving.

    PairMOT tracking validation is launched asynchronously by the evaluator and
    appends its metrics to ``scalars.json`` after the validation epoch returns.
    This hook therefore checks completed tracking rows from previous validation
    epochs. If the latest tracking job finishes after this hook runs, it will be
    consumed at the next validation epoch, which intentionally permits stopping
    one validation later than a synchronous metric would.
    """

    priority = 'LOW'

    def __init__(self,
                 cls_key: str = 'pair/track/cls_hota',
                 det_key: str = 'pair/track/det_hota',
                 min_delta: float = 0.001,
                 patience: int = 4,
                 strict: bool = False,
                 check_finite: bool = True,
                 monitor: Optional[str] = None,
                 rule: Optional[str] = None) -> None:
        self.cls_key = cls_key
        self.det_key = det_key
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.strict = bool(strict)
        self.check_finite = bool(check_finite)
        # Accepted for config compatibility with mmengine EarlyStoppingHook.
        self.monitor = monitor
        self.rule = rule

        self.best_cls = -math.inf
        self.best_det = -math.inf
        self.bad_count = 0
        self._processed_steps = set()

    @staticmethod
    def _latest_scalars_path(work_dir: Optional[str]) -> Optional[str]:
        if not work_dir or not osp.isdir(work_dir):
            return None
        candidates = []
        for name in os.listdir(work_dir):
            path = osp.join(work_dir, name, 'vis_data', 'scalars.json')
            if osp.isfile(path):
                candidates.append(path)
        if not candidates:
            return None
        return max(candidates, key=osp.getmtime)

    @staticmethod
    def _to_float(value) -> Optional[float]:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value

    def _load_new_track_rows(self, scalars_path: str):
        rows = []
        with open(scalars_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if self.cls_key not in row or self.det_key not in row:
                    continue
                step = int(row.get('step', line_idx))
                row_id = (step, line_idx)
                if row_id in self._processed_steps:
                    continue
                cls_hota = self._to_float(row.get(self.cls_key))
                det_hota = self._to_float(row.get(self.det_key))
                if cls_hota is None or det_hota is None:
                    continue
                rows.append((row_id, step, cls_hota, det_hota))
        return rows

    def _is_improved(self, value: float, best: float) -> bool:
        return value > best + self.min_delta

    def _log(self, runner: Runner, message: str) -> None:
        rank, _ = get_dist_info()
        if rank == 0:
            runner.logger.info(message)

    def _process_rows(self,
                      runner: Runner,
                      rows,
                      allow_stop: bool = True) -> None:
        for row_id, step, cls_hota, det_hota in rows:
            self._processed_steps.add(row_id)
            if self.check_finite and (
                    not math.isfinite(cls_hota)
                    or not math.isfinite(det_hota)):
                if allow_stop:
                    runner.train_loop.stop_training = True
                self._log(
                    runner,
                    '[PairTrackEarlyStopping] non-finite tracking metric: '
                    f'step={step}, cls_hota={cls_hota}, det_hota={det_hota}.')
                return

            cls_improved = self._is_improved(cls_hota, self.best_cls)
            det_improved = self._is_improved(det_hota, self.best_det)
            if cls_improved:
                self.best_cls = cls_hota
            if det_improved:
                self.best_det = det_hota

            if cls_improved or det_improved:
                self.bad_count = 0
                status = 'improved'
            else:
                self.bad_count += 1
                status = f'no_improve={self.bad_count}/{self.patience}'

            self._log(
                runner,
                '[PairTrackEarlyStopping] '
                f'step={step} cls_hota={cls_hota:.4f} '
                f'det_hota={det_hota:.4f} best_cls={self.best_cls:.4f} '
                f'best_det={self.best_det:.4f} {status}.')

        if allow_stop and self.bad_count >= self.patience:
            runner.train_loop.stop_training = True
            self._log(
                runner,
                '[PairTrackEarlyStopping] stop training: both cls_hota and '
                f'det_hota have not improved for {self.bad_count} completed '
                'track validations.')

    def before_train(self, runner: Runner) -> None:
        assert hasattr(runner.train_loop, 'stop_training'), (
            '`train_loop` should contain `stop_training` variable.')
        scalars_path = self._latest_scalars_path(getattr(runner, 'work_dir', None))
        if scalars_path is None:
            return
        rows = self._load_new_track_rows(scalars_path)
        if rows:
            self._process_rows(runner, rows, allow_stop=False)
            self._log(
                runner,
                '[PairTrackEarlyStopping] restored state from existing '
                f'track metrics: best_cls={self.best_cls:.4f}, '
                f'best_det={self.best_det:.4f}, '
                f'no_improve={self.bad_count}/{self.patience}.')

    def after_val_epoch(self,
                        runner: Runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        scalars_path = self._latest_scalars_path(getattr(runner, 'work_dir', None))
        if scalars_path is None:
            if self.strict:
                raise RuntimeError(
                    'PairTrackEarlyStoppingHook cannot find scalars.json.')
            self._log(
                runner,
                '[PairTrackEarlyStopping] waiting for scalars.json.')
            return

        rows = self._load_new_track_rows(scalars_path)
        if not rows:
            self._log(
                runner,
                '[PairTrackEarlyStopping] no completed async track metrics '
                'available yet; skip this validation.')
            return

        self._process_rows(runner, rows, allow_stop=True)
