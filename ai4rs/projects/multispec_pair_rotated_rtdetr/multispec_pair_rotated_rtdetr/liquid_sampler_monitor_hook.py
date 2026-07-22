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
            image_variant_ratio = selected.ne(
                selected[:1]).any(dim=-1).any(dim=-1).float().mean()
            canonical = selected.sort(dim=-1).values
            same_set = canonical[:, :, None].eq(
                canonical[:, None, :]).all(dim=-1)
            max_set_repeat = same_set.sum(dim=-1).max(dim=-1).values.float()
            first_occurrence = torch.ones(
                canonical.shape[:2], device=canonical.device,
                dtype=torch.bool)
            for group_idx in range(1, num_groups):
                first_occurrence[:, group_idx] = ~canonical[
                    :, group_idx, None].eq(
                        canonical[:, :group_idx]).all(dim=-1).any(dim=-1)
            unique_sets = first_occurrence.sum(dim=-1).float()
            pair_batch_size = getattr(stem, 'pair_batch_size', None)
            if (pair_batch_size is not None and pair_batch_size > 0
                    and pair_batch_size * 2 == selected.size(0)):
                pair_route_mismatch = selected[:pair_batch_size].ne(
                    selected[pair_batch_size:]).float().mean()
            else:
                pair_route_mismatch = None

            runner.message_hub.update_scalar(
                'liquid/max_prob', float(max_prob.mean().item()))
            runner.message_hub.update_scalar(
                'liquid/entropy', float(entropy.mean().item()))
            runner.message_hub.update_scalar(
                'liquid/changed_ratio', float(changed.item()))
            runner.message_hub.update_scalar(
                'liquid/image_variant_ratio',
                float(image_variant_ratio.item()))
            runner.message_hub.update_scalar(
                'liquid/unique_sets', float(unique_sets.mean().item()))
            runner.message_hub.update_scalar(
                'liquid/max_set_repeat',
                float(max_set_repeat.mean().item()))
            if pair_route_mismatch is not None:
                runner.message_hub.update_scalar(
                    'liquid/pair_route_mismatch',
                    float(pair_route_mismatch.item()))
            set_max_load = getattr(sampler, 'last_set_max_load', None)
            if set_max_load is not None:
                runner.message_hub.update_scalar(
                    'liquid/set_max_load',
                    float(set_max_load.float().mean().item()))
            set_margin = getattr(sampler, 'last_set_margin', None)
            set_diversity_gate = getattr(
                sampler, 'last_set_diversity_gate', None)
            if set_margin is not None:
                runner.message_hub.update_scalar(
                    'liquid/set_margin',
                    float(set_margin.float().mean().item()))
            if set_diversity_gate is not None:
                runner.message_hub.update_scalar(
                    'liquid/set_diversity_gate',
                    float(set_diversity_gate.float().mean().item()))
            consensus = getattr(sampler, 'pair_consensus_router', None)
            consensus_residual = getattr(
                consensus, 'last_pair_residual', None)
            frame_logit_distance = getattr(
                consensus, 'last_frame_logit_distance', None)
            frame_reliability = getattr(
                consensus, 'last_frame_reliability', None)
            if consensus_residual is not None:
                consensus_residual_abs = (
                    consensus_residual.float().abs().mean())
                runner.message_hub.update_scalar(
                    'liquid/consensus_residual_abs',
                    float(consensus_residual_abs.item()))
                runner.message_hub.update_scalar(
                    'liquid/frame_logit_distance',
                    float(frame_logit_distance.float().item()))
                if frame_reliability is not None:
                    reliability_offset = (
                        frame_reliability.float() - 0.5).abs().mean()
                    reliability_entropy = -(
                        frame_reliability.float().clamp_min(1e-6).log() *
                        frame_reliability.float()).sum(dim=0).mean()
                    runner.message_hub.update_scalar(
                        'liquid/frame_reliability_offset',
                        float(reliability_offset.item()))
                    runner.message_hub.update_scalar(
                        'liquid/frame_reliability_entropy',
                        float(reliability_entropy.item()))
            else:
                consensus_residual_abs = None
            competitive = getattr(sampler, 'competitive_router', None)
            common_correction = getattr(
                competitive, 'last_common_correction', None)
            specific_correction = getattr(
                competitive, 'last_specific_correction', None)
            content_score = getattr(
                competitive, 'last_content_score', None)
            anchor_scale = getattr(
                competitive, 'last_anchor_scale', None)
            if common_correction is not None:
                common_abs = common_correction.float().abs().mean()
                specific_abs = specific_correction.float().abs().mean()
                content_sample_std = content_score.float().std(
                    dim=0, unbiased=False).mean()
                runner.message_hub.update_scalar(
                    'liquid/common_abs', float(common_abs.item()))
                runner.message_hub.update_scalar(
                    'liquid/specific_abs', float(specific_abs.item()))
                runner.message_hub.update_scalar(
                    'liquid/content_sample_std',
                    float(content_sample_std.item()))
                anchor_scale_mean = anchor_scale.float().mean()
                anchor_scale_min = anchor_scale.float().amin()
                runner.message_hub.update_scalar(
                    'liquid/anchor_scale_mean',
                    float(anchor_scale_mean.item()))
                runner.message_hub.update_scalar(
                    'liquid/anchor_scale_min',
                    float(anchor_scale_min.item()))
            else:
                common_abs = None
                specific_abs = None
                content_sample_std = None
                anchor_scale_mean = None
                anchor_scale_min = None
            preserving = getattr(
                sampler, 'confidence_preserving_router', None)
            preserving_residual = getattr(
                preserving, 'last_residual', None)
            if preserving_residual is not None:
                preserving_residual_abs = (
                    preserving_residual.float().abs().mean())
                preserving_gate = (
                    preserving.last_uncertainty_gate.float().mean())
                preserving_margin = (
                    preserving.last_task_margin.float().mean())
                preserving_content_std = (
                    preserving.last_content_score.float().std(
                        dim=0, unbiased=False).mean())
                runner.message_hub.update_scalar(
                    'liquid/preserving_residual_abs',
                    float(preserving_residual_abs.item()))
                runner.message_hub.update_scalar(
                    'liquid/preserving_gate', float(preserving_gate.item()))
                runner.message_hub.update_scalar(
                    'liquid/preserving_margin', float(preserving_margin.item()))
                runner.message_hub.update_scalar(
                    'liquid/preserving_content_std',
                    float(preserving_content_std.item()))
            else:
                preserving_residual_abs = None
                preserving_gate = None
                preserving_margin = None
                preserving_content_std = None
            sparse = getattr(sampler, 'sparse_spectral_evidence', None)
            scale_weights = getattr(sparse, 'last_scale_weights', None)
            if scale_weights is not None:
                small_scale_weight = scale_weights.float()[..., 0].mean()
                sparse_router_gain = sparse.router_gain.float().abs().mean()
                sparse_laf_gain = (
                    stem.liquid_aware_fusion.sparse_spatial_gain.float()
                    .abs().mean())
                runner.message_hub.update_scalar(
                    'liquid/sparse_small_scale_weight',
                    float(small_scale_weight.item()))
                runner.message_hub.update_scalar(
                    'liquid/sparse_router_gain',
                    float(sparse_router_gain.item()))
                runner.message_hub.update_scalar(
                    'liquid/sparse_laf_gain',
                    float(sparse_laf_gain.item()))
            else:
                small_scale_weight = None
                sparse_router_gain = None
                sparse_laf_gain = None
            detail_module = getattr(
                stem, 'pair_consistent_detail_preservation', None)
            compact_detail_module = getattr(
                stem, 'pair_aligned_compact_detail_enhancement', None)
            if detail_module is None:
                detail_module = compact_detail_module
            detail_gain = getattr(detail_module, 'last_detail_gain', None)
            if detail_gain is not None:
                detail_gain_abs = detail_gain.float().abs().mean()
                detail_gain_std = detail_gain.float().std(unbiased=False)
                runner.message_hub.update_scalar(
                    'liquid/detail_gain_abs',
                    float(detail_gain_abs.item()))
                runner.message_hub.update_scalar(
                    'liquid/detail_gain_std',
                    float(detail_gain_std.item()))
                compact_mask_mean = getattr(
                    compact_detail_module, 'last_compact_mask_mean', None)
                uncertainty_mean = getattr(
                    compact_detail_module, 'last_uncertainty_mean', None)
                if compact_mask_mean is not None:
                    runner.message_hub.update_scalar(
                        'liquid/compact_mask_mean',
                        float(compact_mask_mean.float().item()))
                    runner.message_hub.update_scalar(
                        'liquid/detail_uncertainty_mean',
                        float(uncertainty_mean.float().item()))
            else:
                detail_gain_abs = None
                detail_gain_std = None
                compact_mask_mean = None
                uncertainty_mean = None
            aligned_delta = getattr(
                getattr(stem, 'liquid_aware_fusion', None),
                'last_pair_aligned_delta', None)
            if aligned_delta is not None:
                aligned_delta_abs = aligned_delta.float().abs().mean()
                runner.message_hub.update_scalar(
                    'liquid/aligned_delta_abs',
                    float(aligned_delta_abs.item()))
            else:
                aligned_delta_abs = None
            quality = getattr(stem, 'fusion_quality_conservation', None)
            quality_shift = getattr(quality, 'last_shift_abs', None)
            quality_response = getattr(
                quality, 'last_response_projection_abs', None)
            quality_error = getattr(quality, 'last_constraint_error', None)
            if quality_shift is not None:
                runner.message_hub.update_scalar(
                    'liquid/quality_shift_abs',
                    float(quality_shift.float().item()))
                runner.message_hub.update_scalar(
                    'liquid/quality_response_projection_abs',
                    float(quality_response.float().item()))
                runner.message_hub.update_scalar(
                    'liquid/quality_constraint_error',
                    float(quality_error.float().item()))

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
            f'hard={sampler.hard} '
            f'max_prob={max_prob.mean().item():.4f} '
            f'entropy={entropy.mean().item():.4f} '
            f'changed_ratio={changed.item():.4f} '
            f'image_variant_ratio={image_variant_ratio.item():.4f} '
            f'unique_sets={unique_sets.mean().item():.2f} '
            f'max_set_repeat={max_set_repeat.mean().item():.2f} '
            f'pair_route_mismatch='
            f'{float(pair_route_mismatch.item()) if pair_route_mismatch is not None else -1.0:.3f} '
            f'set_transport={sampler.set_transport_strength:.3f} '
            f'set_max_load='
            f'{float(set_max_load.float().mean().item()) if set_max_load is not None else 0.0:.3f} '
            f'set_margin='
            f'{float(set_margin.float().mean().item()) if set_margin is not None else 0.0:.3f} '
            f'set_diversity_gate='
            f'{float(set_diversity_gate.float().mean().item()) if set_diversity_gate is not None else 0.0:.3f} '
            f'consensus_residual='
            f'{float(consensus_residual_abs.item()) if consensus_residual_abs is not None else 0.0:.3f} '
            f'common_abs='
            f'{float(common_abs.item()) if common_abs is not None else 0.0:.3f} '
            f'specific_abs='
            f'{float(specific_abs.item()) if specific_abs is not None else 0.0:.3f} '
            f'content_sample_std='
            f'{float(content_sample_std.item()) if content_sample_std is not None else 0.0:.3f} '
            f'anchor_scale='
            f'{float(anchor_scale_mean.item()) if anchor_scale_mean is not None else 1.0:.3f}/'
            f'{float(anchor_scale_min.item()) if anchor_scale_min is not None else 1.0:.3f} '
            f'preserve_residual='
            f'{float(preserving_residual_abs.item()) if preserving_residual_abs is not None else 0.0:.3f} '
            f'preserve_gate='
            f'{float(preserving_gate.item()) if preserving_gate is not None else 0.0:.3f} '
            f'preserve_margin='
            f'{float(preserving_margin.item()) if preserving_margin is not None else 0.0:.3f} '
            f'preserve_content_std='
            f'{float(preserving_content_std.item()) if preserving_content_std is not None else 0.0:.3f} '
            f'sparse_scale3='
            f'{float(small_scale_weight.item()) if small_scale_weight is not None else 0.0:.3f} '
            f'sparse_gain='
            f'{float(sparse_router_gain.item()) if sparse_router_gain is not None else 0.0:.3f}/'
            f'{float(sparse_laf_gain.item()) if sparse_laf_gain is not None else 0.0:.3f} '
            f'detail_gain='
            f'{float(detail_gain_abs.item()) if detail_gain_abs is not None else 0.0:.3f}/'
            f'{float(detail_gain_std.item()) if detail_gain_std is not None else 0.0:.3f} '
            f'compact_detail='
            f'{float(compact_mask_mean.float().item()) if compact_mask_mean is not None else 0.0:.3f}/'
            f'{float(uncertainty_mean.float().item()) if uncertainty_mean is not None else 0.0:.3f} '
            f'aligned_delta='
            f'{float(aligned_delta_abs.item()) if aligned_delta_abs is not None else 0.0:.3f} '
            f'quality_projection='
            f'{float(quality_shift.float().item()) if quality_shift is not None else 0.0:.3f}/'
            f'{float(quality_response.float().item()) if quality_response is not None else 0.0:.3f}/'
            f'{float(quality_error.float().item()) if quality_error is not None else 0.0:.6f} '
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
                 set_transport_start: float | None = None,
                 set_transport_end: float | None = None,
                 set_transport_anneal_epochs: float | None = None,
                 log_interval: int = 200) -> None:
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.anneal_epochs = anneal_epochs
        self.hard_start_epoch = hard_start_epoch
        self.set_transport_start = set_transport_start
        self.set_transport_end = set_transport_end
        self.set_transport_anneal_epochs = set_transport_anneal_epochs
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
        if (self.set_transport_start is not None
                and self.set_transport_end is not None):
            transport_epochs = float(
                self.set_transport_anneal_epochs or anneal_epochs)
            transport_progress = min(max(
                epoch_float / max(transport_epochs, 1e-6), 0.0), 1.0)
            sampler.set_transport_strength = float(
                self.set_transport_start + (
                    self.set_transport_end - self.set_transport_start
                ) * transport_progress)

        if self.log_interval > 0 and self.every_n_train_iters(
                runner, self.log_interval):
            runner.logger.info(
                '[LiquidSamplerAnneal] '
                f'epoch={epoch_float:.3f} tau={sampler.tau:.4f} '
                f'hard={sampler.hard} '
                f'set_transport={sampler.set_transport_strength:.3f}')

    def before_train_epoch(self, runner: Runner) -> None:
        self._set_sampler_state(runner, 0)

    def before_train_iter(self,
                          runner: Runner,
                          batch_idx: int,
                          data_batch=None) -> None:
        self._set_sampler_state(runner, batch_idx)
