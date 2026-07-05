# Copyright (c) AI4RS. All rights reserved.
"""Monitor pair temporal adapter learning signals during training."""

from __future__ import annotations

import torch
from mmengine.dist import get_dist_info
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmrotate.registry import HOOKS


@HOOKS.register_module()
class PairTemporalAdapterMonitorHook(Hook):
    """Log gamma and gradient diagnostics for the P5 temporal adapter."""

    priority = 'ABOVE_NORMAL'

    def __init__(self, interval: int = 50) -> None:
        self.interval = int(interval)
        self._prev_gamma = {}
        self._prev_named_params = {}

    @staticmethod
    def _unwrap_model(runner: Runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        return model

    @staticmethod
    def _find_adapters(model):
        encoder = getattr(model, 'encoder', None)
        if encoder is None:
            return []
        adapters = []
        for name in ('pair_temporal_adapter', 'post_pair_temporal_adapter'):
            adapter = getattr(encoder, name, None)
            if adapter is not None:
                adapters.append((name, adapter))
        return adapters

    @staticmethod
    def _delta_abs_mean(param, prev) -> float:
        if param is None or prev is None:
            return 0.0
        return float((param.detach() - prev).abs().mean().item())

    def _param_delta(self, name: str, param) -> float:
        if param is None:
            return 0.0
        prev = self._prev_named_params.get(name)
        delta = self._delta_abs_mean(param.detach(), prev)
        self._prev_named_params[name] = param.detach().clone()
        return delta

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        adapters = self._find_adapters(self._unwrap_model(runner))
        if not adapters:
            return

        with torch.no_grad():
            log_chunks = []
            for adapter_name, adapter in adapters:
                gamma_tensor = adapter.gamma.detach()
                if gamma_tensor.numel() == 1:
                    gamma = float(gamma_tensor.item())
                    gamma_abs = abs(gamma)
                else:
                    gamma = float(gamma_tensor.mean().item())
                    gamma_abs = float(gamma_tensor.abs().max().item())
                gamma_delta = self._delta_abs_mean(
                    gamma_tensor, self._prev_gamma.get(adapter_name))
                self._prev_gamma[adapter_name] = gamma_tensor.clone()
                prefix = f'pair_temporal/{adapter_name}'
                runner.message_hub.update_scalar(f'{prefix}_gamma', gamma)
                runner.message_hub.update_scalar(
                    f'{prefix}_gamma_abs', gamma_abs)
                runner.message_hub.update_scalar(
                    f'{prefix}_gamma_delta_abs', gamma_delta)
                delta_items = []
                if hasattr(adapter, 'attn'):
                    delta_items.append((
                        'attn',
                        self._param_delta(
                            f'{adapter_name}.attn.in_proj_weight',
                            adapter.attn.in_proj_weight)))
                if hasattr(adapter, 'out_proj'):
                    delta_items.append((
                        'out_proj',
                        self._param_delta(
                            f'{adapter_name}.out_proj.weight',
                            adapter.out_proj.weight)))
                if hasattr(adapter, 'context_mlp'):
                    delta_items.append((
                        'context_mlp',
                        self._param_delta(
                            f'{adapter_name}.context_mlp.0.weight',
                            adapter.context_mlp[0].weight)))
                if hasattr(adapter, 'delta_conv'):
                    delta_items.append((
                        'delta_conv',
                        self._param_delta(
                            f'{adapter_name}.delta_conv.last.weight',
                            adapter.delta_conv[-1].weight)))
                if hasattr(adapter, 'gate_mlps'):
                    delta_items.append((
                        'gate_mlp',
                        self._param_delta(
                            f'{adapter_name}.gate_mlps.0.0.weight',
                            adapter.gate_mlps[0][0].weight)))
                if hasattr(adapter, 'local_blocks'):
                    delta_items.append((
                        'local_block',
                        self._param_delta(
                            f'{adapter_name}.local_blocks.0.last.weight',
                            adapter.local_blocks[0][-1].weight)))
                for name, value in delta_items:
                    runner.message_hub.update_scalar(
                        f'{prefix}_{name}_delta_abs', value)
                log_chunks.append(
                    f'{adapter_name}: gamma={gamma:.6g} '
                    f'|gamma|max={gamma_abs:.6g} '
                    f'gamma_delta={gamma_delta:.6g} '
                    + ' '.join(
                        f'{name}_delta={value:.6g}'
                        for name, value in delta_items))

        if not self.every_n_train_iters(runner, self.interval):
            return
        rank, _ = get_dist_info()
        if rank != 0:
            return
        runner.logger.info(
            '[PairTemporalAdapter] ' + ' ; '.join(log_chunks))
