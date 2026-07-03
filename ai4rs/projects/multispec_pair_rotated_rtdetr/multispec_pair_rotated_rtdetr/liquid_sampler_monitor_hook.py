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

        with torch.no_grad():
            probs = probs.detach()
            eps = torch.finfo(probs.dtype).eps
            max_prob, selected = probs.max(dim=-1)
            entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)
            num_groups = probs.size(1)
            spectral_kernel = probs.size(2)
            fixed = torch.empty(
                num_groups,
                spectral_kernel,
                device=selected.device,
                dtype=selected.dtype)
            for group_idx in range(num_groups):
                for kernel_idx in range(spectral_kernel):
                    fixed[group_idx, kernel_idx] = group_idx + kernel_idx
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
