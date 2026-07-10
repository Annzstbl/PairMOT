# Copyright (c) AI4RS. All rights reserved.
"""Monitor Liquid Spectral Sampling patterns during training."""

from __future__ import annotations

import torch
from mmengine.dist import get_dist_info
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmrotate.registry import HOOKS


@HOOKS.register_module()
class LiquidSamplerMonitorHook(Hook):
    """Log compact diagnostics from the multispectral liquid stem."""

    priority = 'LOW'

    def __init__(self, interval: int = 50) -> None:
        self.interval = int(interval)

    @staticmethod
    def _unwrap_model(runner: Runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        return model

    @staticmethod
    def _find_stem(model):
        backbone = getattr(model, 'backbone', None)
        stem = getattr(backbone, 'stem', None)
        if stem is None or len(stem) == 0:
            return None
        liquid_stem = stem[0]
        if getattr(liquid_stem, 'liquid_sampler', None) is None:
            return None
        return liquid_stem

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        stem = self._find_stem(self._unwrap_model(runner))
        if stem is None:
            return
        probs = getattr(stem, 'last_liquid_probs', None)
        if probs is None:
            return
        indices = getattr(stem, 'last_liquid_indices', None)

        with torch.no_grad():
            probs = probs.detach()
            eps = torch.finfo(probs.dtype).eps
            max_prob, raw_selected = probs.max(dim=-1)
            selected = raw_selected if indices is None else indices.detach()
            entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)
            num_groups = probs.size(1)
            spectral_kernel = probs.size(2)
            sampler = getattr(stem, 'liquid_sampler', None)
            fixed = getattr(sampler, 'init_pattern_indices', None)
            if fixed is None:
                fixed = torch.empty(
                    num_groups,
                    spectral_kernel,
                    device=selected.device,
                    dtype=selected.dtype)
                for group_idx in range(num_groups):
                    for kernel_idx in range(spectral_kernel):
                        fixed[group_idx, kernel_idx] = group_idx + kernel_idx
            else:
                fixed = fixed.to(device=selected.device, dtype=selected.dtype)
            changed = (selected != fixed.unsqueeze(0)).float().mean()

            runner.message_hub.update_scalar(
                'liquid/max_prob', float(max_prob.mean().item()))
            runner.message_hub.update_scalar(
                'liquid/entropy', float(entropy.mean().item()))
            runner.message_hub.update_scalar(
                'liquid/changed_ratio', float(changed.item()))

        if not self.every_n_train_iters(runner, self.interval):
            return
        rank, _ = get_dist_info()
        if rank != 0:
            return

        pattern = selected[0].detach().cpu().tolist()
        pattern_text = ' / '.join(''.join(str(int(x)) for x in group)
                                  for group in pattern)
        runner.logger.info(
            '[LiquidSampler] '
            f'max_prob={max_prob.mean().item():.4f} '
            f'entropy={entropy.mean().item():.4f} '
            f'changed_ratio={changed.item():.4f} '
            f'pattern={pattern_text}')


@HOOKS.register_module()
class LiquidSamplerAnnealHook(Hook):
    """Anneal liquid sampler temperature and optionally enable hard sampling."""

    priority = 'ABOVE_NORMAL'

    def __init__(self,
                 tau_start: float = 2.0,
                 tau_end: float = 0.5,
                 anneal_epochs: float | None = None,
                 hard_start_epoch: float | None = None,
                 log_interval: int = 200) -> None:
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.anneal_epochs = anneal_epochs
        self.hard_start_epoch = hard_start_epoch
        self.log_interval = int(log_interval)

    @staticmethod
    def _unwrap_model(runner: Runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        return model

    @staticmethod
    def _find_sampler(model):
        backbone = getattr(model, 'backbone', None)
        stem = getattr(backbone, 'stem', None)
        if stem is None or len(stem) == 0:
            return None
        return getattr(stem[0], 'liquid_sampler', None)

    @staticmethod
    def _max_epochs(runner: Runner) -> float:
        if getattr(runner, 'max_epochs', None) is not None:
            return float(runner.max_epochs)
        train_loop = getattr(runner, 'train_loop', None)
        if train_loop is not None and getattr(train_loop, 'max_epochs', None):
            return float(train_loop.max_epochs)
        return 1.0

    @staticmethod
    def _iters_per_epoch(runner: Runner) -> int:
        dataloader = getattr(runner, 'train_dataloader', None)
        if dataloader is None:
            return 1
        try:
            return max(1, len(dataloader))
        except TypeError:
            return 1

    def _set_sampler_state(self, runner: Runner, batch_idx: int = 0) -> None:
        sampler = self._find_sampler(self._unwrap_model(runner))
        if sampler is None:
            return

        max_epochs = self._max_epochs(runner)
        anneal_epochs = float(self.anneal_epochs or max_epochs)
        iter_offset = batch_idx / self._iters_per_epoch(runner)
        epoch_float = float(runner.epoch) + iter_offset
        progress = min(max(epoch_float / max(anneal_epochs, 1e-6), 0.0), 1.0)
        sampler.tau = self.tau_start + (
            self.tau_end - self.tau_start) * progress
        if self.hard_start_epoch is not None:
            sampler.hard = epoch_float >= float(self.hard_start_epoch)

        if self.log_interval > 0 and self.every_n_train_iters(
                runner, self.log_interval):
            runner.logger.info(
                '[LiquidSamplerAnneal] '
                f'epoch={epoch_float:.3f} tau={sampler.tau:.4f} '
                f'hard={sampler.hard}')

    def before_train_epoch(self, runner: Runner) -> None:
        self._set_sampler_state(runner, 0)

    def before_train_iter(self,
                          runner: Runner,
                          batch_idx: int,
                          data_batch=None) -> None:
        self._set_sampler_state(runner, batch_idx)
