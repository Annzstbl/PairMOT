# Copyright (c) AI4RS. All rights reserved.
import math
from itertools import combinations, permutations
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrotate.registry import MODELS

# Match ResNetV1d deep-stem first Conv2d: kernel=3, stride=2, padding=1.
STEM_SPATIAL_KERNEL = 3
STEM_SPATIAL_STRIDE = 2
STEM_SPATIAL_PADDING = 1
# Spectral kernel size equals RGB channel count for weight inheritance.
STEM_SPECTRAL_KERNEL = 3


class ScaleAdaptiveSparseSpectralEvidence(nn.Module):
    """Extract shared compact-target evidence for routing and fusion."""

    def __init__(self,
                 num_spectral: int,
                 num_groups: int,
                 spectral_kernel: int,
                 embed_dims: int) -> None:
        super().__init__()
        self.num_spectral = num_spectral
        self.num_groups = num_groups
        self.spectral_kernel = spectral_kernel
        self.scale_router = nn.Linear(4, 2)
        self.evidence_proj = nn.Linear(2, embed_dims)
        self.pair_norm = nn.LayerNorm(8)
        self.pair_proj = nn.Linear(8, embed_dims)
        self.group_slot_query = nn.Parameter(torch.empty(
            num_groups, spectral_kernel, embed_dims))
        self.router_gain = nn.Parameter(torch.zeros(
            num_groups, spectral_kernel, 1))

        nn.init.zeros_(self.scale_router.weight)
        nn.init.zeros_(self.scale_router.bias)
        nn.init.xavier_uniform_(self.evidence_proj.weight)
        nn.init.zeros_(self.evidence_proj.bias)
        nn.init.xavier_uniform_(self.pair_proj.weight)
        nn.init.zeros_(self.pair_proj.bias)
        nn.init.normal_(self.group_slot_query, std=0.02)

        self.last_band_evidence = None
        self.last_band_embedding = None
        self.last_contrast = None
        self.last_scale_weights = None
        self.last_logit_delta = None

    @staticmethod
    def _normalize_bands(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.flatten(2).std(
            dim=-1, unbiased=False).view(*x.shape[:2], 1, 1)
        return (x - mean) / std.clamp_min(1e-4)

    @staticmethod
    def _local_contrast(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        context = F.avg_pool2d(
            x, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2)
        return (x - context).abs()

    @staticmethod
    def _pair_other(value: torch.Tensor,
                    pair_batch_size: Optional[int]) -> torch.Tensor:
        if (pair_batch_size is None or pair_batch_size <= 0
                or pair_batch_size * 2 != value.size(0)):
            return value
        return torch.cat([
            value[pair_batch_size:], value[:pair_batch_size]
        ], dim=0)

    def forward(self, x: torch.Tensor,
                pair_batch_size: Optional[int]) -> torch.Tensor:
        z = self._normalize_bands(x)
        # Match the stem spatial stride so evidence maps can be reused by LAF.
        z = F.avg_pool2d(z, kernel_size=2, stride=2, ceil_mode=True)
        contrast_small = self._local_contrast(z, 3)
        contrast_medium = self._local_contrast(z, 9)
        scale_stats = torch.stack([
            contrast_small.mean(dim=(-2, -1)),
            contrast_small.flatten(2).std(dim=-1, unbiased=False),
            contrast_medium.mean(dim=(-2, -1)),
            contrast_medium.flatten(2).std(dim=-1, unbiased=False),
        ], dim=-1)
        scale_weights = F.softmax(self.scale_router(scale_stats), dim=-1)
        contrast = (
            scale_weights[..., 0, None, None] * contrast_small +
            scale_weights[..., 1, None, None] * contrast_medium)

        saliency = contrast.square().mean(dim=1, keepdim=True).sqrt()
        spatial_weight = saliency / saliency.sum(
            dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        signed_response = (spatial_weight * z).sum(dim=(-2, -1))
        contrast_response = (spatial_weight * contrast).sum(dim=(-2, -1))
        band_evidence = torch.stack([
            signed_response, contrast_response
        ], dim=-1)

        # Normalize across physical bands so a scene-wide response cannot
        # become a common preference shared by every group.
        centered = band_evidence - band_evidence.mean(dim=1, keepdim=True)
        scale = centered.square().mean(
            dim=1, keepdim=True).add(1e-6).sqrt()
        normalized = centered / scale
        other = self._pair_other(normalized, pair_batch_size)
        pair_relation = torch.cat([
            normalized, other, normalized - other, normalized * other
        ], dim=-1)
        band_embedding = (
            self.evidence_proj(normalized) +
            self.pair_proj(self.pair_norm(pair_relation)))
        band_embedding = F.normalize(band_embedding, dim=-1)
        query = F.normalize(self.group_slot_query, dim=-1)
        logit_delta = torch.einsum(
            'gkd,bsd->bgks', query, band_embedding)
        logit_delta = logit_delta - logit_delta.mean(dim=-1, keepdim=True)
        logit_delta = torch.tanh(self.router_gain).unsqueeze(0) * logit_delta

        self.last_band_evidence = normalized
        self.last_band_embedding = band_embedding
        self.last_contrast = contrast
        self.last_scale_weights = scale_weights
        self.last_logit_delta = logit_delta
        return logit_delta

    def group_map(self, probs: torch.Tensor) -> torch.Tensor:
        assert self.last_contrast is not None
        coverage = F.normalize(probs.sum(dim=2), p=1, dim=-1)
        contrast = self.last_contrast.flatten(2)
        group_map = torch.bmm(coverage, contrast).view(
            probs.size(0), self.num_groups,
            *self.last_contrast.shape[-2:])
        return group_map / group_map.mean(
            dim=(-2, -1), keepdim=True).clamp_min(1e-6)


def calc_temporal_output_size(num_spectral: int,
                              temporal_padding: int,
                              temporal_kernel: int,
                              temporal_stride: int) -> int:
    """Compute spectral (temporal) dim size after the 3D stem conv."""
    return (math.floor(
        (num_spectral + 2 * temporal_padding - (temporal_kernel - 1) - 1) /
        temporal_stride) + 1)


def uniform_gate_logit(num_bands: int) -> float:
    """Logit bias so ``sigmoid(x) == 1 / num_bands`` for every band."""
    assert num_bands > 1
    return math.log(1.0 / (num_bands - 1))


class LiquidSpectralSampler(nn.Module):
    """Input-conditioned spectral sampler for 3-band Conv3d windows.

    By default the sampler outputs
    ``num_groups = num_spectral - spectral_kernel + 1`` adjacent groups. A
    custom number of groups and explicit initial patterns can be supplied for
    cyclic or task-specific spectral windows. Soft sampling remains a spectral
    fusion distribution; hard/eval-hard sampling selects bands without
    replacement inside each group. Optionally, hard sampling can also assign
    a distinct unordered band set to every group.
    """

    def __init__(self,
                 num_spectral: int = 8,
                 spectral_kernel: int = STEM_SPECTRAL_KERNEL,
                 embed_dims: int = 32,
                 num_groups: Optional[int] = None,
                 init_patterns: Optional[Sequence[Sequence[int]]] = None,
                 tau: float = 1.0,
                 hard: bool = False,
                 init_logit: float = 8.0,
                 head_weight_std: float = 0.0,
                 deterministic_eval: bool = True,
                 eval_hard: bool = True,
                 lowres_grad_size: Optional[Union[int, Tuple[int, int]]] = None,
                 lowres_grad_downsample: int = 4,
                 use_lowres_grad_correction: bool = True,
                 lowres_grad_upsample_mode: str = 'nearest',
                 use_band_attention: bool = False,
                 band_attention_heads: int = 4,
                 band_attention_dropout: float = 0.0,
                 hard_group_unique_sets: bool = False,
                 use_soft_context_after_hard: bool = False,
                 soft_group_set_transport: Optional[dict] = None,
                 pair_sampler_router: Optional[dict] = None,
                 pair_consensus_router: Optional[dict] = None,
                 pair_band_context: Optional[dict] = None,
                 competitive_router: Optional[dict] = None,
                 confidence_preserving_router: Optional[dict] = None,
                 sparse_spectral_evidence: Optional[dict] = None,
                 block_route_descriptor: Optional[dict] = None,
                 coarse_spectral_preview_router: Optional[dict] = None) -> None:
        super().__init__()
        assert num_spectral >= spectral_kernel
        self.num_spectral = num_spectral
        self.spectral_kernel = spectral_kernel
        self.num_groups = int(num_groups or
                              (num_spectral - spectral_kernel + 1))
        assert self.num_groups > 0
        self.embed_dims = embed_dims
        self.tau = tau
        self.hard = hard
        self.deterministic_eval = deterministic_eval
        self.eval_hard = eval_hard
        self.lowres_grad_size = lowres_grad_size
        self.lowres_grad_downsample = lowres_grad_downsample
        self.use_lowres_grad_correction = use_lowres_grad_correction
        if lowres_grad_upsample_mode not in ('nearest', 'bilinear'):
            raise ValueError(
                'lowres_grad_upsample_mode must be nearest or bilinear, got '
                f'{lowres_grad_upsample_mode!r}')
        self.lowres_grad_upsample_mode = lowres_grad_upsample_mode
        self.use_band_attention = use_band_attention
        self.hard_group_unique_sets = hard_group_unique_sets
        self.use_soft_context_after_hard = use_soft_context_after_hard
        if soft_group_set_transport is True:
            soft_group_set_transport = {}
        transport_cfg = dict(soft_group_set_transport or {})
        self.use_soft_group_set_transport = soft_group_set_transport is not None
        self.set_transport_num_iters = int(
            transport_cfg.get('num_iters', 16))
        self.set_transport_temperature = float(
            transport_cfg.get('temperature', 1.0))
        self.set_transport_strength = float(
            transport_cfg.get('initial_strength', 0.0))
        self.set_transport_confidence_gated = bool(
            transport_cfg.get('confidence_gated', False))
        self.set_transport_margin_threshold = float(
            transport_cfg.get('margin_threshold', 0.35))
        self.set_transport_margin_temperature = float(
            transport_cfg.get('margin_temperature', 0.1))
        self.set_transport_min_gate = float(
            transport_cfg.get('min_gate', 0.05))
        self.set_transport_apply_to_independent_hard = bool(
            transport_cfg.get('apply_to_independent_hard', False))
        assert self.set_transport_num_iters > 0
        assert self.set_transport_temperature > 0
        assert 0.0 <= self.set_transport_strength <= 1.0
        assert self.set_transport_margin_temperature > 0
        assert 0.0 <= self.set_transport_min_gate <= 1.0
        init_pattern_tensor = self._build_init_patterns(init_patterns)
        self.register_buffer(
            'init_pattern_indices', init_pattern_tensor, persistent=False)
        if hard_group_unique_sets or self.use_soft_group_set_transport:
            candidate_sets = list(combinations(
                range(num_spectral), spectral_kernel))
            assert self.num_groups <= len(candidate_sets), (
                'Group-set routing requires at least as many '
                f'band sets as groups, got {len(candidate_sets)} sets for '
                f'{self.num_groups} groups')
            candidate_permutations = [
                list(permutations(candidate_set))
                for candidate_set in candidate_sets
            ]
            self.register_buffer(
                'hard_candidate_permutations',
                torch.tensor(candidate_permutations, dtype=torch.long),
                persistent=False)
            if self.use_soft_group_set_transport:
                candidate_one_hot = F.one_hot(
                    self.hard_candidate_permutations,
                    num_classes=num_spectral).to(torch.float32)
                self.register_buffer(
                    'set_candidate_one_hot',
                    candidate_one_hot,
                    persistent=False)
            else:
                self.set_candidate_one_hot = None
        else:
            self.hard_candidate_permutations = None
            self.set_candidate_one_hot = None
        self.last_hard_indices = None
        self.last_context_probs = None
        self.last_set_assignment = None
        self.last_set_max_load = None
        self.last_set_margin = None
        self.last_set_diversity_gate = None

        if block_route_descriptor is False:
            block_route_descriptor = None
        elif block_route_descriptor is True:
            block_route_descriptor = {}
        block_cfg = dict(block_route_descriptor or {})
        self.use_block_route_descriptor = block_route_descriptor is not None
        self.block_grid_size = tuple(block_cfg.get('grid_size', (12, 16)))
        assert len(self.block_grid_size) == 2
        assert all(int(size) > 0 for size in self.block_grid_size)
        self.block_grid_size = tuple(
            int(size) for size in self.block_grid_size)
        if coarse_spectral_preview_router is False:
            coarse_spectral_preview_router = None
        elif coarse_spectral_preview_router is True:
            coarse_spectral_preview_router = {}
        preview_cfg = dict(coarse_spectral_preview_router or {})
        self.use_coarse_spectral_preview = (
            coarse_spectral_preview_router is not None)
        self.preview_grid_size = tuple(
            int(size) for size in preview_cfg.get('grid_size', (24, 32)))
        self.preview_detach_shared_weight = bool(
            preview_cfg.get('detach_shared_weight', True))
        assert len(self.preview_grid_size) == 2
        assert all(size > 0 for size in self.preview_grid_size)
        assert not (self.use_block_route_descriptor
                    and self.use_coarse_spectral_preview), (
                        'block_route_descriptor and '
                        'coarse_spectral_preview_router are mutually exclusive')
        if self.use_coarse_spectral_preview:
            assert self.preview_detach_shared_weight, (
                'CSPR must detach the shared Conv3D weight so the preview '
                'router cannot perturb the formal stem through a second path')
        if self.use_block_route_descriptor:
            assert pair_band_context is None, (
                'block_route_descriptor does not support pair_band_context; '
                'pair coupling must operate on the aggregated route hidden')
            assert competitive_router is None, (
                'block_route_descriptor and competitive_router are mutually '
                'exclusive')
            assert confidence_preserving_router is None, (
                'block_route_descriptor and confidence_preserving_router are '
                'mutually exclusive')
        if self.use_coarse_spectral_preview:
            assert pair_band_context is None, (
                'coarse_spectral_preview_router does not support '
                'pair_band_context; pair coupling operates on preview hidden')
            assert competitive_router is None, (
                'coarse_spectral_preview_router and competitive_router are '
                'mutually exclusive')
            assert confidence_preserving_router is None, (
                'coarse_spectral_preview_router and '
                'confidence_preserving_router are mutually exclusive')

        self.desc_proj = nn.Linear(3, embed_dims)
        self.band_embedding = nn.Parameter(torch.zeros(num_spectral, embed_dims))
        self.w1 = nn.Linear(embed_dims * 2, embed_dims)
        self.w2 = nn.Linear(embed_dims * 2, embed_dims)
        if self.use_block_route_descriptor:
            self.block_summary_norm = nn.LayerNorm(embed_dims * 3)
            self.block_summary_proj = nn.Linear(embed_dims * 3, embed_dims)
        else:
            self.block_summary_norm = None
            self.block_summary_proj = None
        self.head = nn.Linear(
            embed_dims, self.num_groups * spectral_kernel * num_spectral)
        if pair_sampler_router is True:
            pair_sampler_router = {}
        if pair_consensus_router is True:
            pair_consensus_router = {}
        assert not (pair_sampler_router is not None
                    and pair_consensus_router is not None), (
                        'pair_sampler_router and pair_consensus_router are '
                        'mutually exclusive')
        if pair_sampler_router is not None:
            router_cfg = dict(pair_sampler_router)
            router_cfg.setdefault('embed_dims', embed_dims)
            router_cfg.setdefault(
                'output_dims',
                self.num_groups * spectral_kernel * num_spectral)
            self.pair_sampler_router = PairCoupledSamplerRouter(**router_cfg)
        else:
            self.pair_sampler_router = None
        if pair_consensus_router is not None:
            consensus_cfg = dict(pair_consensus_router)
            consensus_cfg.setdefault('embed_dims', embed_dims)
            consensus_cfg.setdefault('num_spectral', num_spectral)
            consensus_cfg.setdefault(
                'output_dims',
                self.num_groups * spectral_kernel * num_spectral)
            self.pair_consensus_router = PairConsensusSamplerRouter(
                **consensus_cfg)
        else:
            self.pair_consensus_router = None
        if pair_band_context is True:
            pair_band_context = {}
        if pair_band_context is not None:
            context_cfg = dict(pair_band_context)
            context_cfg.setdefault('embed_dims', embed_dims)
            context_cfg.setdefault(
                'output_dims',
                self.num_groups * spectral_kernel * num_spectral)
            self.pair_band_context = PairBandContextEncoder(**context_cfg)
        else:
            self.pair_band_context = None
        if competitive_router is True:
            competitive_router = {}
        if competitive_router is not None:
            competitive_cfg = dict(competitive_router)
            competitive_cfg.setdefault('num_spectral', num_spectral)
            competitive_cfg.setdefault('num_groups', self.num_groups)
            competitive_cfg.setdefault('spectral_kernel', spectral_kernel)
            competitive_cfg.setdefault(
                'init_patterns', init_pattern_tensor.tolist())
            competitive_cfg.setdefault('anchor_strength', init_logit)
            self.competitive_router = AnchorResidualCompetitiveRouter(
                **competitive_cfg)
        else:
            self.competitive_router = None
        if confidence_preserving_router is True:
            confidence_preserving_router = {}
        assert not (competitive_router is not None
                    and confidence_preserving_router is not None), (
                        'competitive_router and confidence_preserving_router '
                        'are mutually exclusive')
        if confidence_preserving_router is not None:
            preserving_cfg = dict(confidence_preserving_router)
            preserving_cfg.setdefault('num_spectral', num_spectral)
            preserving_cfg.setdefault('num_groups', self.num_groups)
            preserving_cfg.setdefault('spectral_kernel', spectral_kernel)
            self.confidence_preserving_router = (
                ConfidencePreservingAdaptiveRouter(**preserving_cfg))
        else:
            self.confidence_preserving_router = None
        if sparse_spectral_evidence is True:
            sparse_spectral_evidence = {}
        if sparse_spectral_evidence is not None:
            sparse_cfg = dict(sparse_spectral_evidence)
            sparse_cfg.setdefault('num_spectral', num_spectral)
            sparse_cfg.setdefault('num_groups', self.num_groups)
            sparse_cfg.setdefault('spectral_kernel', spectral_kernel)
            sparse_cfg.setdefault('embed_dims', embed_dims)
            self.sparse_spectral_evidence = (
                ScaleAdaptiveSparseSpectralEvidence(**sparse_cfg))
        else:
            self.sparse_spectral_evidence = None
        self.last_sparse_band_evidence = None
        self.last_sparse_group_map = None
        self.last_pair_band_context = None
        self.last_pair_band_logits = None
        self.last_block_statistics = None
        self.last_block_hidden = None
        self.last_block_summary = None
        self.last_preview_features = None
        self.last_preview_statistics = None
        if use_band_attention:
            assert embed_dims % band_attention_heads == 0, (
                f'embed_dims={embed_dims} must be divisible by '
                f'band_attention_heads={band_attention_heads}')
            self.band_norm1 = nn.LayerNorm(embed_dims)
            self.band_attn = nn.MultiheadAttention(
                embed_dims,
                band_attention_heads,
                dropout=band_attention_dropout,
                batch_first=True)
            self.band_norm2 = nn.LayerNorm(embed_dims)
            self.band_ffn = nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 2),
                nn.GELU(),
                nn.Dropout(band_attention_dropout),
                nn.Linear(embed_dims * 2, embed_dims),
            )
        else:
            self.band_norm1 = None
            self.band_attn = None
            self.band_norm2 = None
            self.band_ffn = None
        self._init_weights(init_logit, head_weight_std)

    def _build_init_patterns(
            self,
            init_patterns: Optional[Sequence[Sequence[int]]]) -> torch.Tensor:
        if init_patterns is None:
            patterns = []
            for group_idx in range(self.num_groups):
                patterns.append([
                    (group_idx + kernel_idx) % self.num_spectral
                    for kernel_idx in range(self.spectral_kernel)
                ])
        else:
            patterns = [list(group) for group in init_patterns]

        assert len(patterns) == self.num_groups, (
            f'Expected {self.num_groups} initial spectral groups, '
            f'got {len(patterns)}')
        for group in patterns:
            assert len(group) == self.spectral_kernel, (
                f'Each initial group must have {self.spectral_kernel} bands, '
                f'got {len(group)}')
            assert len(set(group)) == len(group), (
                f'Initial liquid spectral group must be unique, got {group}')
            for band_idx in group:
                assert 0 <= int(band_idx) < self.num_spectral, (
                    f'Band index {band_idx} out of range [0, '
                    f'{self.num_spectral})')
        return torch.tensor(patterns, dtype=torch.long)

    def _init_weights(self, init_logit: float, head_weight_std: float) -> None:
        nn.init.zeros_(self.band_embedding)
        nn.init.xavier_uniform_(self.desc_proj.weight)
        nn.init.zeros_(self.desc_proj.bias)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)
        if self.block_summary_proj is not None:
            nn.init.xavier_uniform_(self.block_summary_proj.weight)
            nn.init.zeros_(self.block_summary_proj.bias)
        if self.band_ffn is not None:
            for module in self.band_ffn:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        if head_weight_std > 0:
            nn.init.normal_(self.head.weight, mean=0.0, std=head_weight_std)
        else:
            nn.init.zeros_(self.head.weight)

        bias = torch.zeros(
            self.num_groups,
            self.spectral_kernel,
            self.num_spectral)
        for group_idx in range(self.num_groups):
            for kernel_idx in range(self.spectral_kernel):
                band_idx = self.init_pattern_indices[group_idx, kernel_idx]
                bias[group_idx, kernel_idx, band_idx] = init_logit
        with torch.no_grad():
            self.head.bias.copy_(bias.reshape(-1))

    def _assign_unique_band_sets(self, logits: torch.Tensor) -> torch.Tensor:
        """Assign one distinct unordered band set to every hard group."""
        assert self.hard_candidate_permutations is not None
        candidates = self.hard_candidate_permutations
        num_sets, num_permutations, spectral_kernel = candidates.shape
        batch_size, num_groups = logits.shape[:2]

        expanded_logits = logits[:, :, None, None].expand(
            -1, -1, num_sets, num_permutations, -1, -1)
        gather_indices = candidates.view(
            1, 1, num_sets, num_permutations, spectral_kernel, 1).expand(
                batch_size, num_groups, -1, -1, -1, -1)
        permutation_scores = expanded_logits.gather(
            -1, gather_indices).squeeze(-1).sum(dim=-1)
        set_scores, best_permutation = permutation_scores.max(dim=-1)

        # Regret-first greedy matching avoids fixed group-order priority. With
        # only 8 groups and 56 candidate sets this remains entirely on GPU.
        available_sets = torch.ones(
            batch_size, num_sets, dtype=torch.bool, device=logits.device)
        unassigned_groups = torch.ones(
            batch_size, num_groups, dtype=torch.bool, device=logits.device)
        selected_sets = torch.full(
            (batch_size, num_groups), -1, dtype=torch.long,
            device=logits.device)
        batch_indices = torch.arange(batch_size, device=logits.device)

        for _ in range(num_groups):
            available_scores = set_scores.masked_fill(
                ~available_sets[:, None], -float('inf'))
            available_scores = available_scores.masked_fill(
                ~unassigned_groups[:, :, None], -float('inf'))
            top_scores, top_sets = available_scores.topk(2, dim=-1)
            confidence = top_scores[..., 0] - top_scores[..., 1]
            confidence = confidence.masked_fill(
                ~unassigned_groups, -float('inf'))
            next_group = confidence.argmax(dim=-1)
            next_set = top_sets[batch_indices, next_group, 0]
            selected_sets[batch_indices, next_group] = next_set
            unassigned_groups[batch_indices, next_group] = False
            available_sets[batch_indices, next_set] = False

        group_indices = torch.arange(num_groups, device=logits.device)
        group_indices = group_indices.unsqueeze(0).expand(batch_size, -1)
        selected_permutations = best_permutation[
            batch_indices[:, None], group_indices, selected_sets]
        return candidates[selected_sets, selected_permutations]

    def _project_soft_group_sets(self,
                                 raw_probs: torch.Tensor) -> torch.Tensor:
        """Project slot probabilities onto capacity-limited band sets.

        Slack rows turn the rectangular group-to-set assignment into a square
        transport problem. Each real group keeps unit mass while each
        unordered set has capacity one; unused capacity is absorbed by slack.
        """
        assert self.hard_candidate_permutations is not None
        assert self.set_candidate_one_hot is not None
        candidates = self.hard_candidate_permutations
        num_sets, num_permutations, spectral_kernel = candidates.shape
        batch_size, num_groups = raw_probs.shape[:2]

        # The transport matrix is tiny (at most 56x56), so log-domain FP32
        # gives stable Sinkhorn gradients without touching spatial features.
        log_probs = raw_probs.float().clamp_min(1e-12).log()
        expanded = log_probs[:, :, None, None].expand(
            -1, -1, num_sets, num_permutations, -1, -1)
        gather_indices = candidates.view(
            1, 1, num_sets, num_permutations, spectral_kernel, 1).expand(
                batch_size, num_groups, -1, -1, -1, -1)
        permutation_log_mass = expanded.gather(
            -1, gather_indices).squeeze(-1).sum(dim=-1)
        set_log_mass = torch.logsumexp(permutation_log_mass, dim=-1)
        permutation_probs = F.softmax(permutation_log_mass, dim=-1)

        real_scores = set_log_mass / self.set_transport_temperature
        detached_scores = real_scores.detach()
        task_scale = detached_scores.std(
            dim=-1, unbiased=False).clamp_min(1e-6)
        top2 = detached_scores.topk(2, dim=-1).values
        self.last_set_margin = (
            (top2[..., 0] - top2[..., 1]) / task_scale)
        num_slack = num_sets - num_groups
        if num_slack > 0:
            slack_scores = real_scores.new_zeros(
                batch_size, num_slack, num_sets)
            transport_logits = torch.cat([real_scores, slack_scores], dim=1)
        else:
            transport_logits = real_scores

        for _ in range(self.set_transport_num_iters):
            transport_logits = transport_logits - torch.logsumexp(
                transport_logits, dim=-1, keepdim=True)
            transport_logits = transport_logits - torch.logsumexp(
                transport_logits, dim=-2, keepdim=True)
        transport_logits = transport_logits - torch.logsumexp(
            transport_logits, dim=-1, keepdim=True)
        set_assignment = transport_logits[:, :num_groups].exp()

        candidate_one_hot = self.set_candidate_one_hot.to(
            device=raw_probs.device)
        projected = torch.einsum(
            'bgs,bgsp,spkc->bgkc',
            set_assignment,
            permutation_probs,
            candidate_one_hot)
        projected = projected / projected.sum(dim=-1, keepdim=True).clamp_min(
            1e-12)
        self.last_set_assignment = set_assignment.detach()
        self.last_set_max_load = set_assignment.sum(dim=1).amax(
            dim=-1).detach()
        return projected.to(dtype=raw_probs.dtype)

    def _apply_soft_group_set_transport(
            self, raw_probs: torch.Tensor) -> torch.Tensor:
        if (not self.use_soft_group_set_transport
                or self.set_transport_strength <= 0.0):
            self.last_set_assignment = None
            self.last_set_max_load = None
            self.last_set_margin = None
            self.last_set_diversity_gate = None
            return raw_probs
        projected = self._project_soft_group_sets(raw_probs)
        strength = min(max(self.set_transport_strength, 0.0), 1.0)
        if self.set_transport_confidence_gated:
            assert self.last_set_margin is not None
            uncertainty = torch.sigmoid(
                (self.set_transport_margin_threshold - self.last_set_margin) /
                self.set_transport_margin_temperature)
            gate = (self.set_transport_min_gate +
                    (1.0 - self.set_transport_min_gate) * uncertainty)
            self.last_set_diversity_gate = gate.detach()
            strength_tensor = (strength * gate).unsqueeze(-1).unsqueeze(-1)
            probs = raw_probs + strength_tensor * (projected - raw_probs)
        else:
            self.last_set_diversity_gate = None
            probs = torch.lerp(raw_probs, projected, strength)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(probs.dtype).eps)

    def _dedup_hard_indices(self, logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.hard_group_unique_sets:
                return self._assign_unique_band_sets(logits.detach())
            masked_logits = logits.detach().clone()
            selected = []
            for kernel_idx in range(self.spectral_kernel):
                indices = masked_logits[:, :, kernel_idx].argmax(dim=-1)
                selected.append(indices)
                if kernel_idx + 1 < self.spectral_kernel:
                    for next_idx in range(kernel_idx + 1,
                                          self.spectral_kernel):
                        masked_logits[:, :, next_idx].scatter_(
                            -1, indices.unsqueeze(-1), -float('inf'))
            return torch.stack(selected, dim=-1)

    def _sample_hard_unique(self, logits: torch.Tensor) -> torch.Tensor:
        if self.hard_group_unique_sets:
            if self.training or not self.deterministic_eval:
                eps = torch.finfo(logits.dtype).eps
                uniform = torch.rand_like(logits).clamp_(eps, 1 - eps)
                gumbel = -torch.log(-torch.log(uniform))
                sample_logits = (logits + gumbel) / self.tau
            else:
                sample_logits = logits / self.tau
            soft_probs = F.softmax(sample_logits, dim=-1)
            soft_probs = self._apply_soft_group_set_transport(soft_probs)
            with torch.no_grad():
                indices = self._assign_unique_band_sets(sample_logits)
            self.last_hard_indices = indices
            hard_probs = torch.zeros_like(soft_probs).scatter_(
                -1, indices.unsqueeze(-1), 1.0)
            probs = hard_probs - soft_probs.detach() + soft_probs
            self.last_context_probs = (
                soft_probs if self.use_soft_context_after_hard else probs)
            return probs

        if (self.use_soft_group_set_transport
                and self.set_transport_apply_to_independent_hard):
            if self.training or not self.deterministic_eval:
                eps = torch.finfo(logits.dtype).eps
                uniform = torch.rand_like(logits).clamp_(eps, 1 - eps)
                gumbel = -torch.log(-torch.log(uniform))
                sample_logits = (logits + gumbel) / self.tau
            else:
                sample_logits = logits / self.tau
            soft_probs = F.softmax(sample_logits, dim=-1)
            soft_probs = self._apply_soft_group_set_transport(soft_probs)
            indices = self._dedup_hard_indices(
                soft_probs.float().clamp_min(1e-12).log())
            self.last_hard_indices = indices
            hard_probs = torch.zeros_like(soft_probs).scatter_(
                -1, indices.unsqueeze(-1), 1.0)
            probs = hard_probs - soft_probs.detach() + soft_probs
            self.last_context_probs = (
                soft_probs if self.use_soft_context_after_hard else probs)
            return probs

        if self.training or not self.deterministic_eval:
            masked_logits = logits.clone()
            hard_probs = []
            for kernel_idx in range(self.spectral_kernel):
                probs = F.gumbel_softmax(
                    masked_logits[:, :, kernel_idx],
                    tau=self.tau,
                    hard=True,
                    dim=-1)
                hard_probs.append(probs)
                if kernel_idx + 1 < self.spectral_kernel:
                    indices = probs.detach().argmax(dim=-1)
                    for next_idx in range(kernel_idx + 1,
                                          self.spectral_kernel):
                        masked_logits[:, :, next_idx].scatter_(
                            -1, indices.unsqueeze(-1), -float('inf'))
            probs = torch.stack(hard_probs, dim=2)
            self.last_hard_indices = probs.detach().argmax(dim=-1)
            self.last_context_probs = probs
            return probs

        probs = F.softmax(logits / self.tau, dim=-1)
        indices = self._dedup_hard_indices(logits)
        self.last_hard_indices = indices
        hard_probs = torch.zeros_like(probs).scatter_(
            -1, indices.unsqueeze(-1), 1.0)
        probs = hard_probs - probs.detach() + probs
        self.last_context_probs = probs
        return probs

    @staticmethod
    def _repeat_pair_state(value: Optional[torch.Tensor]
                           ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        return torch.cat([value, value], dim=0)

    def _sample_pair_consensus(self, logits: torch.Tensor,
                               pair_batch_size: int) -> torch.Tensor:
        pair_probs = self._sample(logits[:pair_batch_size])
        self.last_hard_indices = self._repeat_pair_state(
            self.last_hard_indices)
        self.last_context_probs = self._repeat_pair_state(
            self.last_context_probs)
        self.last_set_assignment = self._repeat_pair_state(
            self.last_set_assignment)
        self.last_set_max_load = self._repeat_pair_state(
            self.last_set_max_load)
        self.last_set_margin = self._repeat_pair_state(self.last_set_margin)
        self.last_set_diversity_gate = self._repeat_pair_state(
            self.last_set_diversity_gate)
        return torch.cat([pair_probs, pair_probs], dim=0)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        self.last_hard_indices = self._dedup_hard_indices(logits)
        if self.training or not self.deterministic_eval:
            if self.hard:
                return self._sample_hard_unique(logits)
            probs = F.gumbel_softmax(
                logits, tau=self.tau, hard=self.hard, dim=-1)
            probs = self._apply_soft_group_set_transport(probs)
            self.last_context_probs = probs
            return probs

        probs = F.softmax(logits / self.tau, dim=-1)
        if not self.eval_hard:
            probs = self._apply_soft_group_set_transport(probs)
            self.last_context_probs = probs
            return probs
        return self._sample_hard_unique(logits)

    def _lowres_size(self, height: int, width: int) -> Tuple[int, int]:
        if self.lowres_grad_size is None:
            downsample = max(1, self.lowres_grad_downsample)
            return max(1, height // downsample), max(1, width // downsample)
        if isinstance(self.lowres_grad_size, int):
            size = self.lowres_grad_size
            return min(size, height), min(size, width)
        return min(self.lowres_grad_size[0], height), min(
            self.lowres_grad_size[1], width)

    @staticmethod
    def _bilinear_expand(x: torch.Tensor,
                         output_size: Tuple[int, int]) -> torch.Tensor:
        """Bilinear upsample with activation math kept in ``x.dtype``.

        This reproduces the half-pixel coordinates used by
        ``F.interpolate(..., align_corners=False)`` without invoking the CUDA
        bilinear kernel, which is unavailable for BF16 in PyTorch 2.0.
        """
        input_h, input_w = x.shape[-2:]
        output_h, output_w = output_size
        if (input_h, input_w) == (output_h, output_w):
            return x

        def indices_and_weight(input_size: int, output_size: int):
            position = ((torch.arange(
                output_size, device=x.device, dtype=torch.float32) + 0.5) *
                        (input_size / output_size) - 0.5)
            position = position.clamp_(0, input_size - 1)
            lower = position.floor().to(torch.long)
            upper = (lower + 1).clamp_max_(input_size - 1)
            weight = (position - lower).to(dtype=x.dtype)
            return lower, upper, weight

        h0, h1, hw = indices_and_weight(input_h, output_h)
        expanded = x.index_select(-2, h0)
        expanded = torch.lerp(
            expanded,
            x.index_select(-2, h1),
            hw.view(*([1] * (x.ndim - 2)), output_h, 1))

        w0, w1, ww = indices_and_weight(input_w, output_w)
        return torch.lerp(
            expanded.index_select(-1, w0),
            expanded.index_select(-1, w1),
            ww.view(*([1] * (x.ndim - 1)), output_w))

    def _sample_bands(self, x: torch.Tensor,
                      probs: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        flat_probs = probs.reshape(
            batch_size, self.num_groups * self.spectral_kernel,
            self.num_spectral)

        sampled = torch.bmm(flat_probs.detach(), x.flatten(2)).view(
            batch_size, self.num_groups, self.spectral_kernel, height, width)

        if (not self.training or not self.use_lowres_grad_correction
                or not probs.requires_grad):
            return sampled

        lowres_h, lowres_w = self._lowres_size(height, width)
        lowres_x = F.adaptive_avg_pool2d(
            x.detach(), output_size=(lowres_h, lowres_w))
        lowres_sampled = torch.bmm(flat_probs, lowres_x.flatten(2)).view(
            batch_size, self.num_groups * self.spectral_kernel, lowres_h,
            lowres_w)
        lowres_correction = lowres_sampled - lowres_sampled.detach()
        if self.lowres_grad_upsample_mode == 'bilinear':
            correction = self._bilinear_expand(
                lowres_correction, (height, width))
        else:
            height_indices = torch.div(
                torch.arange(height, device=x.device) * lowres_h,
                height,
                rounding_mode='floor')
            width_indices = torch.div(
                torch.arange(width, device=x.device) * lowres_w,
                width,
                rounding_mode='floor')
            correction = lowres_correction.index_select(
                -2, height_indices).index_select(-1, width_indices)
        correction = correction.view(
                batch_size, self.num_groups, self.spectral_kernel, height,
                width)
        return sampled + correction

    def _spectral_recurrent(self, desc: torch.Tensor) -> torch.Tensor:
        """Aggregate one ordered spectral sequence per batch entry."""
        hidden = desc.new_zeros(desc.size(0), self.embed_dims)
        for band_idx in range(self.num_spectral):
            cell_input = torch.cat([desc[:, band_idx], hidden], dim=-1)
            h_hat = torch.tanh(self.w1(cell_input))
            alpha = torch.sigmoid(self.w2(cell_input))
            hidden = alpha * hidden + (1 - alpha) * h_hat
        return hidden

    def _block_statistics(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-band mean/std/max for every spatial grid block."""
        batch_size, num_spectral, height, width = x.shape
        grid_h, grid_w = self.block_grid_size
        if height % grid_h == 0 and width % grid_w == 0:
            block_h = height // grid_h
            block_w = width // grid_w
            blocks = x.reshape(
                batch_size, num_spectral, grid_h, block_h, grid_w, block_w)
            variance, mean = torch.var_mean(
                blocks, dim=(3, 5), unbiased=False)
            maxv = blocks.amax(dim=(3, 5))
            std = variance.clamp_min(0).sqrt()
        else:
            output_size = (grid_h, grid_w)
            mean = F.adaptive_avg_pool2d(x, output_size)
            second_moment = F.adaptive_avg_pool2d(x.square(), output_size)
            std = (second_moment - mean.square()).clamp_min(0).sqrt()
            maxv = F.adaptive_max_pool2d(x, output_size)
        return torch.stack([mean, std, maxv], dim=-1).flatten(2, 3)

    def _block_route_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Build one global route state from local spectral sequences."""
        block_stats = self._block_statistics(x)
        batch_size, _, num_blocks, _ = block_stats.shape
        block_desc = (
            self.desc_proj(block_stats) +
            self.band_embedding.view(1, self.num_spectral, 1,
                                     self.embed_dims))
        block_desc = block_desc.permute(0, 2, 1, 3).reshape(
            batch_size * num_blocks, self.num_spectral, self.embed_dims)
        if self.band_attn is not None:
            attn_input = self.band_norm1(block_desc)
            block_desc = block_desc + self.band_attn(
                attn_input, attn_input, attn_input)[0]
            block_desc = block_desc + self.band_ffn(
                self.band_norm2(block_desc))

        block_hidden = self._spectral_recurrent(block_desc).view(
            batch_size, num_blocks, self.embed_dims)
        block_summary = torch.cat([
            block_hidden.mean(dim=1),
            block_hidden.std(dim=1, unbiased=False),
            block_hidden.amax(dim=1),
        ], dim=-1)
        hidden = torch.tanh(self.block_summary_proj(
            self.block_summary_norm(block_summary)))

        self.last_block_statistics = block_stats.detach()
        self.last_block_hidden = block_hidden.detach()
        self.last_block_summary = block_summary.detach()
        return hidden

    def _coarse_preview_route_hidden(
            self, x: torch.Tensor,
            preview_weight: Optional[torch.Tensor]) -> torch.Tensor:
        """Build the route from a detached low-resolution Conv3D preview."""
        assert preview_weight is not None, (
            'CSPR requires the formal stem Conv3D weight')
        assert preview_weight.ndim == 5
        assert preview_weight.size(1) == 1
        assert preview_weight.size(2) == self.spectral_kernel

        preview_x = F.adaptive_avg_pool2d(
            x.detach(), self.preview_grid_size)
        cyclic_x = torch.cat(
            [preview_x[:, -1:], preview_x, preview_x[:, :1]], dim=1)
        preview = F.conv3d(
            cyclic_x.unsqueeze(1),
            preview_weight.detach(),
            bias=None,
            stride=1,
            padding=(0, preview_weight.size(-2) // 2,
                     preview_weight.size(-1) // 2))
        reduce_dims = (1, 3, 4)
        preview_mean = preview.mean(dim=reduce_dims)
        preview_rms = preview.square().mean(
            dim=reduce_dims).clamp_min(0).sqrt()
        preview_peak = preview.abs().amax(dim=reduce_dims)
        preview_stats = torch.stack(
            [preview_mean, preview_rms, preview_peak], dim=-1)

        desc = self.desc_proj(preview_stats) + self.band_embedding.unsqueeze(0)
        if self.band_attn is not None:
            attn_input = self.band_norm1(desc)
            desc = desc + self.band_attn(
                attn_input, attn_input, attn_input)[0]
            desc = desc + self.band_ffn(self.band_norm2(desc))
        hidden = self._spectral_recurrent(desc)
        self.last_preview_features = preview.detach()
        self.last_preview_statistics = preview_stats.detach()
        return hidden

    def forward(self,
                x: torch.Tensor,
                pair_batch_size: Optional[int] = None,
                preview_weight: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim == 4, f'Expected [B, S, H, W], got {tuple(x.shape)}'
        assert x.size(1) == self.num_spectral, (
            f'Expected {self.num_spectral} spectral bands, got {x.size(1)}')

        if self.use_block_route_descriptor:
            raw_desc = None
            hidden = self._block_route_hidden(x)
            self.last_pair_band_context = None
            self.last_pair_band_logits = None
            self.last_preview_features = None
            self.last_preview_statistics = None
        elif self.use_coarse_spectral_preview:
            raw_desc = None
            hidden = self._coarse_preview_route_hidden(x, preview_weight)
            self.last_pair_band_context = None
            self.last_pair_band_logits = None
            self.last_block_statistics = None
            self.last_block_hidden = None
            self.last_block_summary = None
        else:
            mean = x.mean(dim=(-2, -1))
            std = x.flatten(2).std(dim=-1)
            maxv = x.amax(dim=(-2, -1))
            raw_desc = torch.stack([mean, std, maxv], dim=-1)
            desc = self.desc_proj(raw_desc) + self.band_embedding.unsqueeze(0)
            if self.band_attn is not None:
                attn_input = self.band_norm1(desc)
                desc = desc + self.band_attn(
                    attn_input, attn_input, attn_input)[0]
                desc = desc + self.band_ffn(self.band_norm2(desc))
            if self.pair_band_context is not None:
                (desc, self.last_pair_band_context,
                 self.last_pair_band_logits) = self.pair_band_context(
                     desc, pair_batch_size)
            else:
                self.last_pair_band_context = None
                self.last_pair_band_logits = None
            hidden = self._spectral_recurrent(desc)
            self.last_block_statistics = None
            self.last_block_hidden = None
            self.last_block_summary = None
            self.last_preview_features = None
            self.last_preview_statistics = None

        logits = self.head(hidden).view(
            x.size(0), self.num_groups, self.spectral_kernel,
            self.num_spectral)
        if self.pair_sampler_router is not None:
            pair_logits = self.pair_sampler_router(hidden, pair_batch_size)
            logits = logits + pair_logits.view_as(logits)
        if self.last_pair_band_logits is not None:
            logits = logits + self.last_pair_band_logits.view_as(logits)
        if self.competitive_router is not None:
            assert raw_desc is not None
            logits = self.competitive_router(
                logits, raw_desc, pair_batch_size)
        if self.confidence_preserving_router is not None:
            assert raw_desc is not None
            logits = self.confidence_preserving_router(
                logits, raw_desc, pair_batch_size)
        if self.sparse_spectral_evidence is not None:
            logits = logits + self.sparse_spectral_evidence(
                x, pair_batch_size)
        if self.pair_consensus_router is not None:
            logits = self.pair_consensus_router(
                logits, hidden, pair_batch_size)
        if (self.pair_consensus_router is not None
                and pair_batch_size is not None and pair_batch_size > 0
                and pair_batch_size * 2 == x.size(0)):
            probs = self._sample_pair_consensus(logits, pair_batch_size)
        else:
            probs = self._sample(logits)
        if self.sparse_spectral_evidence is not None:
            self.last_sparse_band_evidence = (
                self.sparse_spectral_evidence.last_band_evidence)
            self.last_sparse_group_map = (
                self.sparse_spectral_evidence.group_map(probs))
        else:
            self.last_sparse_band_evidence = None
            self.last_sparse_group_map = None
        sampled = self._sample_bands(x, probs)
        return sampled, probs


class PairCoupledSamplerRouter(nn.Module):
    """Condition frame-specific sampler logits on the paired frame.

    The router does not force the two frames to use identical groups.  It
    predicts a residual for each direction from source, paired, difference,
    and agreement features, preserving frame-specific spectral evidence.
    """

    def __init__(self,
                 embed_dims: int,
                 output_dims: int,
                 hidden_dims: int = 64,
                 init_std: float = 1e-3,
                 zero_init: bool = True,
                 relation_mode: str = 'pair_diff_product') -> None:
        super().__init__()
        assert relation_mode in ('pair', 'pair_diff_product')
        self.relation_mode = relation_mode
        relation_dims = embed_dims * (2 if relation_mode == 'pair' else 4)
        self.norm = nn.LayerNorm(relation_dims)
        self.mlp = nn.Sequential(
            nn.Linear(relation_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, output_dims),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
        else:
            nn.init.normal_(self.mlp[-1].weight, std=init_std)
        nn.init.zeros_(self.mlp[-1].bias)

    def _pair_features(self, src: torch.Tensor,
                       other: torch.Tensor) -> torch.Tensor:
        if self.relation_mode == 'pair':
            return torch.cat([src, other], dim=-1)
        return torch.cat([src, other, src - other, src * other], dim=-1)

    def forward(self, hidden: torch.Tensor,
                pair_batch_size: Optional[int]) -> torch.Tensor:
        batch_size = hidden.size(0)
        output_dims = self.mlp[-1].out_features
        if (pair_batch_size is None or pair_batch_size <= 0
                or pair_batch_size * 2 != batch_size):
            return hidden.new_zeros(batch_size, output_dims)

        prev = hidden[:pair_batch_size]
        curr = hidden[pair_batch_size:]
        pair_features = torch.cat([
            self._pair_features(prev, curr),
            self._pair_features(curr, prev),
        ], dim=0)
        return self.mlp(self.norm(pair_features))


class PairConsensusSamplerRouter(nn.Module):
    """Build one spectral route from both frames of an ordered pair.

    Log-mean-exp retains evidence that is strong in either frame. A symmetric,
    zero-initialized residual can then refine the shared route without giving
    either temporal direction a privileged spectral coordinate system.
    """

    def __init__(self,
                 embed_dims: int,
                 output_dims: int,
                 num_spectral: int = 8,
                 hidden_dims: int = 64,
                 reliability_weighted: bool = False,
                 zero_init: bool = True,
                 init_std: float = 1e-3) -> None:
        super().__init__()
        assert output_dims % num_spectral == 0
        self.output_dims = output_dims
        self.num_spectral = num_spectral
        self.num_route_slots = output_dims // num_spectral
        self.reliability_weighted = reliability_weighted
        self.norm = nn.LayerNorm(embed_dims * 3)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dims * 3, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, output_dims),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
        else:
            nn.init.normal_(self.mlp[-1].weight, std=init_std)
        nn.init.zeros_(self.mlp[-1].bias)
        if reliability_weighted:
            # A shared quality estimator gives each frame a group/slot score.
            # Pairwise softmax then makes the aggregation invariant to temporal
            # order. Zero initialization exactly recovers equal log-mean-exp.
            self.reliability_norm = nn.LayerNorm(embed_dims)
            self.reliability_head = nn.Sequential(
                nn.Linear(embed_dims, hidden_dims),
                nn.GELU(),
                nn.Linear(hidden_dims, self.num_route_slots),
            )
            nn.init.xavier_uniform_(self.reliability_head[0].weight)
            nn.init.zeros_(self.reliability_head[0].bias)
            nn.init.zeros_(self.reliability_head[-1].weight)
            nn.init.zeros_(self.reliability_head[-1].bias)
        else:
            self.reliability_norm = None
            self.reliability_head = None
        self.last_pair_residual = None
        self.last_frame_logit_distance = None
        self.last_frame_reliability = None

    def _aggregate_pair_logits(self, prev_logits: torch.Tensor,
                               curr_logits: torch.Tensor,
                               prev_hidden: torch.Tensor,
                               curr_hidden: torch.Tensor) -> torch.Tensor:
        pair_logits = torch.stack([
            prev_logits.float(), curr_logits.float()
        ], dim=0)
        if self.reliability_head is None:
            self.last_frame_reliability = None
            return (torch.logsumexp(pair_logits, dim=0) -
                    math.log(2.0)).to(dtype=prev_logits.dtype)

        quality = torch.stack([
            self.reliability_head(self.reliability_norm(prev_hidden)),
            self.reliability_head(self.reliability_norm(curr_hidden)),
        ], dim=0)
        reliability = F.softmax(quality.float(), dim=0)
        reliability = reliability.view(
            2, prev_logits.size(0), *prev_logits.shape[1:-1], 1)
        shared = torch.logsumexp(
            pair_logits + reliability.clamp_min(1e-6).log(), dim=0)
        self.last_frame_reliability = reliability.detach()
        return shared.to(dtype=prev_logits.dtype)

    def forward(self, logits: torch.Tensor, hidden: torch.Tensor,
                pair_batch_size: Optional[int]) -> torch.Tensor:
        batch_size = logits.size(0)
        if (pair_batch_size is None or pair_batch_size <= 0
                or pair_batch_size * 2 != batch_size):
            self.last_pair_residual = None
            self.last_frame_logit_distance = None
            self.last_frame_reliability = None
            return logits

        prev_logits = logits[:pair_batch_size]
        curr_logits = logits[pair_batch_size:]
        prev_hidden = hidden[:pair_batch_size]
        curr_hidden = hidden[pair_batch_size:]
        shared_task = self._aggregate_pair_logits(
            prev_logits, curr_logits, prev_hidden, curr_hidden)
        pair_feature = torch.cat([
            0.5 * (prev_hidden + curr_hidden),
            (prev_hidden - curr_hidden).abs(),
            prev_hidden * curr_hidden,
        ], dim=-1)
        residual = self.mlp(self.norm(pair_feature)).view_as(shared_task)
        shared_logits = shared_task + residual

        self.last_pair_residual = residual.detach()
        self.last_frame_logit_distance = (
            prev_logits - curr_logits).abs().mean().detach()
        return torch.cat([shared_logits, shared_logits], dim=0)


class AnchorResidualCompetitiveRouter(nn.Module):
    """Keep Liquid routing adaptive without allowing common-mode collapse.

    The fixed cyclic anchors provide spectral coverage. Learned logits are
    decomposed into a bounded common correction and a higher-capacity,
    zero-mean group-specific residual. A separate content path derives
    pair-aware band keys only from per-image statistics, so dataset-level
    biases cannot replace all input-conditioned routing evidence.
    """

    def __init__(self,
                 num_spectral: int,
                 num_groups: int,
                 spectral_kernel: int,
                 init_patterns: Sequence[Sequence[int]],
                 anchor_strength: float = 2.0,
                 content_dims: int = 24,
                 content_strength: float = 0.35,
                 common_cap: float = 0.5,
                 specific_cap: float = 2.0,
                 adaptive_anchor_relax: Optional[dict] = None) -> None:
        super().__init__()
        assert content_dims > 0
        assert content_strength > 0
        assert common_cap > 0
        assert specific_cap > 0
        self.num_spectral = num_spectral
        self.num_groups = num_groups
        self.spectral_kernel = spectral_kernel
        self.content_strength = content_strength
        self.common_cap = common_cap
        self.specific_cap = specific_cap
        relax_cfg = dict(adaptive_anchor_relax or {})
        self.use_adaptive_anchor_relax = adaptive_anchor_relax is not None
        self.max_anchor_relax = float(relax_cfg.get('max_relax', 0.45))
        self.relax_threshold = float(
            relax_cfg.get('evidence_threshold', 0.08))
        self.relax_temperature = float(relax_cfg.get('temperature', 0.02))
        assert 0.0 <= self.max_anchor_relax < 1.0
        assert self.relax_threshold >= 0.0
        assert self.relax_temperature > 0.0

        anchor = torch.zeros(
            num_groups, spectral_kernel, num_spectral)
        for group_idx, pattern in enumerate(init_patterns):
            for slot_idx, band_idx in enumerate(pattern):
                anchor[group_idx, slot_idx, int(band_idx)] = anchor_strength
        self.register_buffer('anchor_logits', anchor, persistent=True)

        # The same content encoder is shared by all physical bands. It has no
        # band or group bias; static specialization lives only in the queries.
        self.content_encoder = nn.Sequential(
            nn.Linear(12, content_dims),
            nn.GELU(),
            nn.Linear(content_dims, content_dims),
        )
        self.group_slot_query = nn.Parameter(torch.empty(
            num_groups, spectral_kernel, content_dims))
        nn.init.xavier_uniform_(self.content_encoder[0].weight)
        nn.init.zeros_(self.content_encoder[0].bias)
        nn.init.xavier_uniform_(self.content_encoder[2].weight)
        nn.init.zeros_(self.content_encoder[2].bias)
        nn.init.normal_(self.group_slot_query, std=0.02)

        self.last_common_correction = None
        self.last_specific_correction = None
        self.last_content_score = None
        self.last_anchor_scale = None

    @staticmethod
    def _normalize_stats(stats: torch.Tensor) -> torch.Tensor:
        centered = stats - stats.mean(dim=1, keepdim=True)
        scale = centered.square().mean(
            dim=1, keepdim=True).add(1e-6).sqrt()
        return centered / scale

    def _pair_content(self, stats: torch.Tensor,
                      pair_batch_size: Optional[int]) -> torch.Tensor:
        stats = self._normalize_stats(stats)
        batch_size = stats.size(0)
        if (pair_batch_size is None or pair_batch_size <= 0
                or pair_batch_size * 2 != batch_size):
            other = stats
        else:
            other = torch.cat([
                stats[pair_batch_size:], stats[:pair_batch_size]
            ], dim=0)
        relation = torch.cat([
            stats, other, stats - other, stats * other
        ], dim=-1)
        return self.content_encoder(relation)

    @staticmethod
    def _bounded(value: torch.Tensor, cap: float) -> torch.Tensor:
        return cap * torch.tanh(value / cap)

    def _adaptive_anchor_scale(
            self, learned_specific: torch.Tensor,
            content_delta: torch.Tensor) -> torch.Tensor:
        if not self.use_adaptive_anchor_relax:
            return learned_specific.new_ones(
                *learned_specific.shape[:-1], 1)

        # Relax an anchor only when task gradients and per-image spectral
        # evidence agree. Weak or opposing evidence leaves the stable anchor
        # untouched instead of manufacturing route variation from noise.
        alignment = F.cosine_similarity(
            learned_specific.float(), content_delta.float(), dim=-1
        ).to(dtype=learned_specific.dtype).clamp_min(0).unsqueeze(-1)
        learned_evidence = learned_specific.square().mean(
            dim=-1, keepdim=True).sqrt()
        confidence = torch.sigmoid(
            (learned_evidence - self.relax_threshold) /
            self.relax_temperature)
        relaxation = self.max_anchor_relax * alignment * confidence
        return 1.0 - relaxation

    def forward(self, logits: torch.Tensor, stats: torch.Tensor,
                pair_batch_size: Optional[int]) -> torch.Tensor:
        anchor = self.anchor_logits.to(dtype=logits.dtype)
        dynamic = logits - anchor.unsqueeze(0)
        common = dynamic.mean(dim=1, keepdim=True)
        specific = dynamic - common

        content_key = F.normalize(
            self._pair_content(stats, pair_batch_size), dim=-1)
        query = F.normalize(self.group_slot_query, dim=-1)
        content_score = torch.einsum(
            'bsc,gkc->bgks', content_key, query)
        content_score = content_score - content_score.mean(
            dim=-1, keepdim=True)
        content_score = content_score - content_score.mean(
            dim=1, keepdim=True)
        content_delta = self.content_strength * content_score
        anchor_scale = self._adaptive_anchor_scale(specific, content_delta)
        specific = specific + content_delta

        common = self._bounded(common, self.common_cap)
        specific = self._bounded(specific, self.specific_cap)
        self.last_common_correction = common.detach()
        self.last_specific_correction = specific.detach()
        self.last_content_score = content_score.detach()
        self.last_anchor_scale = anchor_scale.detach()
        return anchor.unsqueeze(0) * anchor_scale + common + specific


class ConfidencePreservingAdaptiveRouter(nn.Module):
    """Add pair-conditioned evidence only where task routing is uncertain.

    Unlike an anchor router, this module never decomposes, clips, or replaces
    the detector-trained logits. Its band- and group-centered residual can
    reorder close candidates, while a detached top-two margin suppresses the
    residual on confident decisions. This keeps content adaptation from
    overriding task-optimal spectral groups.
    """

    def __init__(self,
                 num_spectral: int,
                 num_groups: int,
                 spectral_kernel: int,
                 content_dims: int = 24,
                 content_strength: float = 0.2,
                 margin_threshold: float = 0.35,
                 margin_temperature: float = 0.1,
                 min_gate: float = 0.05,
                 min_task_scale: float = 0.25,
                 max_task_scale: float = 2.0) -> None:
        super().__init__()
        assert content_dims > 0
        assert content_strength > 0
        assert margin_temperature > 0
        assert 0.0 <= min_gate <= 1.0
        assert 0 < min_task_scale <= max_task_scale
        self.num_spectral = num_spectral
        self.num_groups = num_groups
        self.spectral_kernel = spectral_kernel
        self.content_strength = content_strength
        self.margin_threshold = margin_threshold
        self.margin_temperature = margin_temperature
        self.min_gate = min_gate
        self.min_task_scale = min_task_scale
        self.max_task_scale = max_task_scale

        self.content_encoder = nn.Sequential(
            nn.Linear(12, content_dims, bias=False),
            nn.GELU(),
            nn.Linear(content_dims, content_dims, bias=False),
        )
        self.group_slot_query = nn.Parameter(torch.empty(
            num_groups, spectral_kernel, content_dims))
        nn.init.xavier_uniform_(self.content_encoder[0].weight)
        nn.init.xavier_uniform_(self.content_encoder[2].weight)
        nn.init.normal_(self.group_slot_query, std=0.02)

        self.last_content_score = None
        self.last_uncertainty_gate = None
        self.last_residual = None
        self.last_task_margin = None

    @staticmethod
    def _normalize_stats(stats: torch.Tensor) -> torch.Tensor:
        centered = stats - stats.mean(dim=1, keepdim=True)
        scale = centered.square().mean(
            dim=1, keepdim=True).add(1e-6).sqrt()
        return centered / scale

    def _pair_content(self, stats: torch.Tensor,
                      pair_batch_size: Optional[int]) -> torch.Tensor:
        stats = self._normalize_stats(stats)
        batch_size = stats.size(0)
        if (pair_batch_size is None or pair_batch_size <= 0
                or pair_batch_size * 2 != batch_size):
            other = stats
        else:
            other = torch.cat([
                stats[pair_batch_size:], stats[:pair_batch_size]
            ], dim=0)
        relation = torch.cat([
            stats, other, stats - other, stats * other
        ], dim=-1)
        return self.content_encoder(relation)

    def forward(self, logits: torch.Tensor, stats: torch.Tensor,
                pair_batch_size: Optional[int]) -> torch.Tensor:
        content_key = F.normalize(
            self._pair_content(stats, pair_batch_size), dim=-1)
        query = F.normalize(self.group_slot_query, dim=-1)
        content_score = torch.einsum(
            'bsc,gkc->bgks', content_key, query)
        content_score = content_score - content_score.mean(
            dim=-1, keepdim=True)
        content_score = content_score - content_score.mean(
            dim=1, keepdim=True)

        detached_logits = logits.detach().float()
        raw_task_scale = detached_logits.std(
            dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
        top2 = detached_logits.topk(2, dim=-1).values
        margin = ((top2[..., 0] - top2[..., 1])
                  / raw_task_scale.squeeze(-1)).to(logits.dtype)
        uncertainty = torch.sigmoid(
            (self.margin_threshold - margin) / self.margin_temperature)
        gate = self.min_gate + (1.0 - self.min_gate) * uncertainty
        task_scale = raw_task_scale.clamp(
            self.min_task_scale, self.max_task_scale).to(logits.dtype)
        residual = (self.content_strength * gate.unsqueeze(-1)
                    * task_scale * content_score)

        self.last_content_score = content_score.detach()
        self.last_uncertainty_gate = gate.detach()
        self.last_residual = residual.detach()
        self.last_task_margin = margin.detach()
        return logits + residual


class PairBandContextEncoder(nn.Module):
    """Build a shared pair context for each physically aligned band."""

    def __init__(self,
                 embed_dims: int,
                 output_dims: int,
                 hidden_dims: int = 64,
                 init_std: float = 1e-3,
                 zero_init: bool = True,
                 relation_mode: str = 'pair_diff_product') -> None:
        super().__init__()
        assert relation_mode in ('pair', 'pair_diff_product')
        self.relation_mode = relation_mode
        relation_dims = embed_dims * (2 if relation_mode == 'pair' else 4)
        self.relation_norm = nn.LayerNorm(relation_dims)
        self.context_mlp = nn.Sequential(
            nn.Linear(relation_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, embed_dims),
        )
        self.context_norm = nn.LayerNorm(embed_dims)
        self.desc_delta = nn.Linear(embed_dims, embed_dims)
        self.logit_delta = nn.Linear(embed_dims, output_dims)
        nn.init.xavier_uniform_(self.context_mlp[0].weight)
        nn.init.zeros_(self.context_mlp[0].bias)
        nn.init.xavier_uniform_(self.context_mlp[-1].weight)
        nn.init.zeros_(self.context_mlp[-1].bias)
        if zero_init:
            nn.init.zeros_(self.desc_delta.weight)
            nn.init.zeros_(self.logit_delta.weight)
        else:
            nn.init.normal_(self.desc_delta.weight, std=init_std)
            nn.init.normal_(self.logit_delta.weight, std=init_std)
        nn.init.zeros_(self.desc_delta.bias)
        nn.init.zeros_(self.logit_delta.bias)

    def _relation(self, src: torch.Tensor,
                  other: torch.Tensor) -> torch.Tensor:
        if self.relation_mode == 'pair':
            return torch.cat([src, other], dim=-1)
        common = 0.5 * (src + other)
        return torch.cat([src, common, src - other, src * other], dim=-1)

    def forward(self, desc: torch.Tensor,
                pair_batch_size: Optional[int]
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                           Optional[torch.Tensor]]:
        batch_size = desc.size(0)
        if (pair_batch_size is None or pair_batch_size <= 0
                or pair_batch_size * 2 != batch_size):
            return desc, None, None

        prev = desc[:pair_batch_size]
        curr = desc[pair_batch_size:]
        relation = torch.cat([
            self._relation(prev, curr),
            self._relation(curr, prev),
        ], dim=0)
        context = self.context_mlp(self.relation_norm(relation))
        context = self.context_norm(context)
        pair_logits = self.logit_delta(context.mean(dim=1))
        return desc + self.desc_delta(context), context, pair_logits


class PairTransportTokenCoupling(nn.Module):
    """Align paired group tokens by their sampled spectral coverage."""

    def __init__(self,
                 embed_dims: int,
                 hidden_dims: int = 128,
                 temperature: float = 0.25,
                 init_std: float = 1e-3,
                 zero_init: bool = True,
                 relation_mode: str = 'pair_diff_product') -> None:
        super().__init__()
        assert temperature > 0
        assert relation_mode in ('pair', 'pair_diff_product')
        self.temperature = temperature
        self.relation_mode = relation_mode
        relation_dims = embed_dims * (2 if relation_mode == 'pair' else 4)
        self.norm = nn.LayerNorm(relation_dims)
        self.mlp = nn.Sequential(
            nn.Linear(relation_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, embed_dims),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
        else:
            nn.init.normal_(self.mlp[-1].weight, std=init_std)
        nn.init.zeros_(self.mlp[-1].bias)

    def _relation(self, src: torch.Tensor,
                  transported: torch.Tensor) -> torch.Tensor:
        if self.relation_mode == 'pair':
            return torch.cat([src, transported], dim=-1)
        return torch.cat([
            src, transported, src - transported, src * transported
        ], dim=-1)

    def _transport(self, src_token: torch.Tensor, src_coverage: torch.Tensor,
                   other_token: torch.Tensor,
                   other_coverage: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        affinity = torch.bmm(src_coverage, other_coverage.transpose(1, 2))
        transport = F.softmax(affinity / self.temperature, dim=-1)
        transported = torch.bmm(transport, other_token)
        relation = self.norm(self._relation(src_token, transported))
        return self.mlp(relation), transport

    def forward(self, token: torch.Tensor, probs: torch.Tensor,
                pair_batch_size: Optional[int]
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = token.size(0)
        if (pair_batch_size is None or pair_batch_size <= 0
                or pair_batch_size * 2 != batch_size):
            return torch.zeros_like(token), None

        coverage = F.normalize(probs.sum(dim=2), p=1, dim=-1)
        prev_token = token[:pair_batch_size]
        curr_token = token[pair_batch_size:]
        prev_coverage = coverage[:pair_batch_size]
        curr_coverage = coverage[pair_batch_size:]
        prev_delta, prev_transport = self._transport(
            prev_token, prev_coverage, curr_token, curr_coverage)
        curr_delta, curr_transport = self._transport(
            curr_token, curr_coverage, prev_token, prev_coverage)
        delta = torch.cat([prev_delta, curr_delta], dim=0)
        transport = torch.stack([prev_transport, curr_transport], dim=1)
        return delta, transport


class PairAlignedTokenCoupling(nn.Module):
    """Couple the same group index when paired frames share one route."""

    def __init__(self,
                 embed_dims: int,
                 hidden_dims: int = 128,
                 init_std: float = 1e-3,
                 zero_init: bool = True,
                 relation_mode: str = 'pair_diff_product') -> None:
        super().__init__()
        assert relation_mode in ('pair', 'pair_diff_product')
        self.relation_mode = relation_mode
        relation_dims = embed_dims * (2 if relation_mode == 'pair' else 4)
        self.norm = nn.LayerNorm(relation_dims)
        self.mlp = nn.Sequential(
            nn.Linear(relation_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, embed_dims),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
        else:
            nn.init.normal_(self.mlp[-1].weight, std=init_std)
        nn.init.zeros_(self.mlp[-1].bias)
        self.last_pair_token_distance = None

    def _relation(self, src: torch.Tensor,
                  other: torch.Tensor) -> torch.Tensor:
        if self.relation_mode == 'pair':
            return torch.cat([src, other], dim=-1)
        return torch.cat([src, other, src - other, src * other], dim=-1)

    def forward(self, token: torch.Tensor,
                pair_batch_size: Optional[int]) -> torch.Tensor:
        batch_size = token.size(0)
        if (pair_batch_size is None or pair_batch_size <= 0
                or pair_batch_size * 2 != batch_size):
            self.last_pair_token_distance = None
            return torch.zeros_like(token)

        prev = token[:pair_batch_size]
        curr = token[pair_batch_size:]
        pair_relation = torch.cat([
            self._relation(prev, curr),
            self._relation(curr, prev),
        ], dim=0)
        self.last_pair_token_distance = (
            prev - curr).abs().mean().detach()
        return self.mlp(self.norm(pair_relation))


class PairBandContextFusion(nn.Module):
    """Pool aligned pair-band context into liquid group tokens."""

    def __init__(self,
                 context_dims: int,
                 embed_dims: int,
                 hidden_dims: int = 64,
                 init_std: float = 1e-3,
                 zero_init: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(context_dims)
        self.mlp = nn.Sequential(
            nn.Linear(context_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, embed_dims),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
        else:
            nn.init.normal_(self.mlp[-1].weight, std=init_std)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, pair_band_context: Optional[torch.Tensor],
                probs: torch.Tensor) -> Optional[torch.Tensor]:
        if pair_band_context is None:
            return None
        coverage = F.normalize(probs.sum(dim=2), p=1, dim=-1)
        group_context = torch.bmm(coverage, pair_band_context)
        return self.mlp(self.norm(group_context))


class PairChangeGatedTokenCoupling(nn.Module):
    """Fuse stable pair evidence while preserving frame-specific changes.

    The reliability gate uses only per-group spectral coverage and pooled
    response statistics.  Cross-frame computation is therefore linear in the
    number of liquid groups and never touches full-resolution feature maps.
    """

    def __init__(self,
                 embed_dims: int,
                 hidden_dims: int = 16,
                 init_std: float = 1e-3,
                 zero_init: bool = True) -> None:
        super().__init__()
        self.common_norm = nn.LayerNorm(embed_dims)
        self.change_norm = nn.LayerNorm(embed_dims)
        self.common_proj = nn.Linear(embed_dims, embed_dims)
        self.change_proj = nn.Linear(embed_dims, embed_dims)
        self.gate_mlp = nn.Sequential(
            nn.Linear(4, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, 1),
        )
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        for projection in (self.common_proj, self.change_proj):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        nn.init.xavier_uniform_(self.gate_mlp[0].weight)
        nn.init.zeros_(self.gate_mlp[0].bias)
        nn.init.xavier_uniform_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
        else:
            nn.init.normal_(self.out_proj.weight, std=init_std)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _relative_difference(src: torch.Tensor,
                             other: torch.Tensor) -> torch.Tensor:
        return (src - other).abs() / (
            src.abs() + other.abs()).clamp_min(1e-6)

    def forward(self, token: torch.Tensor, response: torch.Tensor,
                probs: torch.Tensor, pair_batch_size: Optional[int]
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = token.size(0)
        if (pair_batch_size is None or pair_batch_size <= 0
                or pair_batch_size * 2 != batch_size):
            return torch.zeros_like(token), None

        coverage = F.normalize(probs.sum(dim=2), p=1, dim=-1)
        prev_token = token[:pair_batch_size]
        curr_token = token[pair_batch_size:]
        prev_coverage = coverage[:pair_batch_size]
        curr_coverage = coverage[pair_batch_size:]
        prev_response = response[:pair_batch_size]
        curr_response = response[pair_batch_size:]

        coverage_intersection = torch.minimum(
            prev_coverage, curr_coverage).sum(dim=-1, keepdim=True)
        coverage_distance = 0.5 * (
            prev_coverage - curr_coverage).abs().sum(dim=-1, keepdim=True)
        response_difference = self._relative_difference(
            prev_response, curr_response)
        cues = torch.cat([
            coverage_intersection, coverage_distance, response_difference
        ], dim=-1)
        reliability = torch.sigmoid(self.gate_mlp(cues))

        common = 0.5 * (prev_token + curr_token)
        prev_change = prev_token - curr_token
        curr_change = -prev_change
        shared = self.common_proj(self.common_norm(common))

        def _directional_delta(change: torch.Tensor) -> torch.Tensor:
            specific = self.change_proj(self.change_norm(change))
            candidate = reliability * shared + (1.0 - reliability) * specific
            return self.out_proj(F.gelu(candidate))

        delta = torch.cat([
            _directional_delta(prev_change),
            _directional_delta(curr_change),
        ], dim=0)
        pair_reliability = torch.cat([reliability, reliability], dim=0)
        return delta, pair_reliability


class FusionQualityConservation(nn.Module):
    """Project LAF corrections away from global fusion-quality drift.

    Liquid remains free to redistribute evidence between groups. Only the
    correction components that change total SE mass or its response-correlated
    moment are removed, using detached local geometry and no learned loss.
    """

    def __init__(self,
                 mode: str = 'gate_mass',
                 eps: float = 1e-4) -> None:
        super().__init__()
        assert mode in ('gate_mass', 'response_mass', 'dual_moment')
        assert eps > 0
        self.mode = mode
        self.eps = float(eps)
        self.last_shift_abs = None
        self.last_response_projection_abs = None
        self.last_constraint_error = None

    def forward(self, base_logits: torch.Tensor, delta: torch.Tensor,
                response: torch.Tensor) -> torch.Tensor:
        assert base_logits.shape == delta.shape == response.shape
        with torch.no_grad():
            base_gate = base_logits.sigmoid()
            sensitivity = base_gate * (1.0 - base_gate)
            response_scale = response.abs()
            response_scale = response_scale / response_scale.mean(
                dim=1, keepdim=True).clamp_min(self.eps)

            if self.mode == 'response_mass':
                weight = sensitivity * response_scale
            else:
                weight = sensitivity
            weight = weight.clamp_min(self.eps)

        denominator = weight.sum(dim=1, keepdim=True).clamp_min(self.eps)
        shift = (weight * delta).sum(dim=1, keepdim=True) / denominator
        corrected = delta - shift
        response_projection = torch.zeros_like(shift)

        if self.mode == 'dual_moment':
            response_mean = (
                weight * response_scale).sum(dim=1, keepdim=True) / denominator
            response_centered = response_scale - response_mean
            projection_denominator = (
                weight * response_centered.square()).sum(
                    dim=1, keepdim=True).clamp_min(self.eps)
            response_projection = (
                weight * response_centered * corrected).sum(
                    dim=1, keepdim=True) / projection_denominator
            corrected = corrected - response_projection * response_centered

        with torch.no_grad():
            error = (weight * corrected).sum(
                dim=1, keepdim=True).abs() / denominator
            self.last_shift_abs = shift.detach().abs().mean()
            self.last_response_projection_abs = (
                response_projection.detach().abs().mean())
            self.last_constraint_error = error.detach().mean()
        return corrected


class LiquidAwareFusion(nn.Module):
    """Generate SE logit residuals from liquid sampling patterns.

    The branch sees both the conv3d group response and the source-band
    distribution ``P``. Pattern tokens communicate across groups, so the gate
    can react to coverage shifts and duplicated spectral emphasis.
    """

    def __init__(self,
                 num_groups: int,
                 num_spectral: int,
                 spectral_kernel: int,
                 embed_dims: int = 32,
                 num_heads: int = 4,
                 spatial_kernel: int = 3,
                 dropout: float = 0.0,
                 init_std: float = 1e-3,
                 use_overlap_context: bool = False,
                 use_spatial_mixer: bool = True,
                 pair_transport: Optional[dict] = None,
                 pair_aligned_coupling: Optional[dict] = None,
                 pair_band_context_fusion: Optional[dict] = None,
                 pair_change_gate: Optional[dict] = None,
                 use_sparse_evidence: bool = False) -> None:
        super().__init__()
        assert embed_dims > 0
        assert embed_dims % num_heads == 0, (
            f'embed_dims={embed_dims} must be divisible by '
            f'num_heads={num_heads}')
        self.num_groups = num_groups
        self.num_spectral = num_spectral
        self.spectral_kernel = spectral_kernel
        self.use_overlap_context = use_overlap_context
        self.use_spatial_mixer = use_spatial_mixer
        self.use_sparse_evidence = use_sparse_evidence

        pattern_dims = spectral_kernel * num_spectral
        self.pattern_proj = nn.Linear(pattern_dims, embed_dims)
        self.response_proj = nn.Linear(2, embed_dims)
        if use_overlap_context:
            self.overlap_proj = nn.Linear(embed_dims, embed_dims)
        else:
            self.overlap_proj = None
        self.group_embedding = nn.Parameter(torch.zeros(num_groups, embed_dims))
        if use_sparse_evidence:
            self.sparse_evidence_proj = nn.Linear(2, embed_dims)
            self.sparse_spatial_gain = nn.Parameter(torch.zeros(num_groups))
        else:
            self.sparse_evidence_proj = None
            self.sparse_spatial_gain = None
        self.norm1 = nn.LayerNorm(embed_dims)
        self.attn = nn.MultiheadAttention(
            embed_dims,
            num_heads,
            dropout=dropout,
            batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * 2, embed_dims),
        )
        self.token_to_bias = nn.Linear(embed_dims, 1)
        self.token_to_gain = nn.Linear(embed_dims, 1)
        if use_spatial_mixer:
            self.spatial_mixer = nn.Conv2d(
                num_groups,
                num_groups,
                kernel_size=spatial_kernel,
                padding=spatial_kernel // 2,
                bias=True)
        else:
            self.spatial_mixer = None
        self.out_proj = nn.Conv2d(num_groups, num_groups, kernel_size=1)
        if pair_transport is True:
            pair_transport = {}
        if pair_aligned_coupling is True:
            pair_aligned_coupling = {}
        assert not (pair_transport is not None
                    and pair_aligned_coupling is not None), (
                        'pair_transport and pair_aligned_coupling are '
                        'mutually exclusive')
        if pair_transport is not None:
            pair_transport_cfg = dict(pair_transport)
            pair_transport_cfg.setdefault('embed_dims', embed_dims)
            self.pair_transport = PairTransportTokenCoupling(
                **pair_transport_cfg)
        else:
            self.pair_transport = None
        self.last_pair_transport = None
        if pair_aligned_coupling is not None:
            aligned_cfg = dict(pair_aligned_coupling)
            aligned_cfg.setdefault('embed_dims', embed_dims)
            self.pair_aligned_coupling = PairAlignedTokenCoupling(
                **aligned_cfg)
        else:
            self.pair_aligned_coupling = None
        self.last_pair_aligned_delta = None
        if pair_band_context_fusion is True:
            pair_band_context_fusion = {}
        if pair_band_context_fusion is not None:
            pair_band_fusion_cfg = dict(pair_band_context_fusion)
            pair_band_fusion_cfg.setdefault('embed_dims', embed_dims)
            self.pair_band_context_fusion = PairBandContextFusion(
                **pair_band_fusion_cfg)
        else:
            self.pair_band_context_fusion = None
        if pair_change_gate is True:
            pair_change_gate = {}
        if pair_change_gate is not None:
            pair_change_gate_cfg = dict(pair_change_gate)
            pair_change_gate_cfg.setdefault('embed_dims', embed_dims)
            self.pair_change_gate = PairChangeGatedTokenCoupling(
                **pair_change_gate_cfg)
        else:
            self.pair_change_gate = None
        self.last_pair_change_reliability = None
        self._init_weights(init_std)

    def _init_weights(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.group_embedding, std=init_std)
        nn.init.xavier_uniform_(self.pattern_proj.weight)
        nn.init.zeros_(self.pattern_proj.bias)
        nn.init.xavier_uniform_(self.response_proj.weight)
        nn.init.zeros_(self.response_proj.bias)
        if self.sparse_evidence_proj is not None:
            nn.init.zeros_(self.sparse_evidence_proj.weight)
            nn.init.zeros_(self.sparse_evidence_proj.bias)
        if self.overlap_proj is not None:
            nn.init.xavier_uniform_(self.overlap_proj.weight)
            nn.init.zeros_(self.overlap_proj.bias)
        if self.spatial_mixer is not None:
            nn.init.xavier_uniform_(self.spatial_mixer.weight)
            nn.init.zeros_(self.spatial_mixer.bias)
        nn.init.normal_(self.token_to_bias.weight, std=init_std)
        nn.init.zeros_(self.token_to_bias.bias)
        nn.init.normal_(self.token_to_gain.weight, std=init_std)
        nn.init.zeros_(self.token_to_gain.bias)
        nn.init.normal_(self.out_proj.weight, std=init_std)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self,
                x_se: torch.Tensor,
                probs: torch.Tensor,
                pair_batch_size: Optional[int] = None,
                pair_band_context: Optional[torch.Tensor] = None,
                sparse_band_evidence: Optional[torch.Tensor] = None,
                sparse_group_map: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        batch_size, num_groups, height, width = x_se.shape
        assert num_groups == self.num_groups
        pattern = probs.reshape(batch_size, num_groups, -1)
        spatial_mean = x_se.mean(dim=(-2, -1))
        spatial_std = x_se.flatten(2).std(dim=-1)
        response = torch.stack([spatial_mean, spatial_std], dim=-1)

        token = (self.pattern_proj(pattern) + self.response_proj(response) +
                 self.group_embedding.unsqueeze(0))
        if self.sparse_evidence_proj is not None:
            assert sparse_band_evidence is not None
            coverage = F.normalize(probs.sum(dim=2), p=1, dim=-1)
            group_evidence = torch.bmm(coverage, sparse_band_evidence)
            token = token + self.sparse_evidence_proj(group_evidence)
        if self.overlap_proj is not None:
            coverage = probs.sum(dim=2)
            coverage = F.normalize(coverage, p=1, dim=-1)
            overlap = torch.bmm(coverage, coverage.transpose(1, 2))
            overlap = overlap / overlap.sum(dim=-1, keepdim=True).clamp_min(
                1e-6)
            token = token + self.overlap_proj(torch.bmm(overlap, token))
        if self.pair_band_context_fusion is not None:
            pair_band_delta = self.pair_band_context_fusion(
                pair_band_context, probs)
            if pair_band_delta is not None:
                token = token + pair_band_delta
        if self.pair_change_gate is not None:
            pair_change_delta, self.last_pair_change_reliability = (
                self.pair_change_gate(
                    token, response, probs, pair_batch_size))
            token = token + pair_change_delta
        else:
            self.last_pair_change_reliability = None
        if self.pair_transport is not None:
            pair_delta, self.last_pair_transport = self.pair_transport(
                token, probs, pair_batch_size)
            token = token + pair_delta
        else:
            self.last_pair_transport = None
        if self.pair_aligned_coupling is not None:
            self.last_pair_aligned_delta = self.pair_aligned_coupling(
                token, pair_batch_size)
            token = token + self.last_pair_aligned_delta
        else:
            self.last_pair_aligned_delta = None
        attn_input = self.norm1(token)
        token = token + self.attn(attn_input, attn_input, attn_input)[0]
        token = token + self.ffn(self.norm2(token))

        pattern_bias = self.token_to_bias(token).transpose(1, 2).view(
            batch_size, 1, num_groups, 1, 1)
        pattern_gain = self.token_to_gain(token).transpose(1, 2).view(
            batch_size, 1, num_groups, 1, 1)

        if self.spatial_mixer is not None:
            spatial = self.spatial_mixer(x_se).unsqueeze(1)
            delta = torch.tanh(pattern_gain) * spatial + pattern_bias
            delta = delta.squeeze(1)
            if self.sparse_spatial_gain is not None:
                assert sparse_group_map is not None
                if sparse_group_map.shape[-2:] != (height, width):
                    sparse_group_map = F.interpolate(
                        sparse_group_map, size=(height, width),
                        mode='nearest')
                local_gain = torch.tanh(
                    self.sparse_spatial_gain).view(1, num_groups, 1, 1)
                delta = delta + local_gain * sparse_group_map * spatial.squeeze(1)
        else:
            delta = pattern_bias.squeeze(1).expand(
                batch_size, num_groups, height, width)
        return self.out_proj(F.gelu(delta))


class LiquidGroupModulator(nn.Module):
    """Reweight liquid conv3d groups from sampling coverage descriptors."""

    def __init__(self,
                 num_groups: int,
                 num_spectral: int,
                 spectral_kernel: int,
                 hidden_dims: int = 16,
                 init_std: float = 1e-3) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.num_spectral = num_spectral
        self.spectral_kernel = spectral_kernel
        self.group_embedding = nn.Parameter(torch.zeros(num_groups, hidden_dims))
        self.mlp = nn.Sequential(
            nn.Linear(num_spectral + 3, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, 1),
        )
        self._init_weights(init_std)

    def _init_weights(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.group_embedding, std=init_std)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.mlp[-1].weight, std=init_std)

    def forward(self, x: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_groups, height, width = x.shape
        assert num_groups == self.num_groups
        coverage = probs.sum(dim=2)
        coverage = coverage / coverage.sum(dim=-1, keepdim=True).clamp_min(
            1e-6)
        entropy = -(coverage.clamp_min(1e-6) *
                    coverage.clamp_min(1e-6).log()).sum(dim=-1, keepdim=True)
        entropy = entropy / math.log(self.num_spectral)
        peak = coverage.amax(dim=-1, keepdim=True)
        response = x.detach().abs().mean(dim=(1, 3, 4), keepdim=False)
        response = response.unsqueeze(-1)
        descriptor = torch.cat([coverage, entropy, peak, response], dim=-1)
        hidden = self.mlp[0](descriptor) + self.group_embedding.unsqueeze(0)
        hidden = self.mlp[1](hidden)
        gain = self.mlp[2](hidden).view(batch_size, 1, num_groups, 1, 1)
        return x * (1.0 + torch.tanh(gain))


class PairConsistentDetailPreservation(nn.Module):
    """Restore compact spatial detail suppressed by spectral SE fusion.

    The spatial mask is frame-specific, while its group gain uses paired
    summary statistics and spectral-coverage agreement. This avoids assuming
    pixel alignment between moving objects in adjacent frames.
    """

    def __init__(self,
                 num_groups: int,
                 num_spectral: int,
                 hidden_dims: int = 16) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.num_spectral = num_spectral
        self.group_embedding = nn.Parameter(torch.zeros(
            num_groups, hidden_dims))
        self.gain_mlp = nn.Sequential(
            nn.Linear(10, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, 1),
        )
        nn.init.normal_(self.group_embedding, std=0.02)
        nn.init.xavier_uniform_(self.gain_mlp[0].weight)
        nn.init.zeros_(self.gain_mlp[0].bias)
        # Exact identity at initialization while retaining a gradient path.
        nn.init.zeros_(self.gain_mlp[-1].weight)
        nn.init.zeros_(self.gain_mlp[-1].bias)
        self.last_detail_gain = None
        self.last_detail_mask_mean = None

    @staticmethod
    def _pair_other(value: torch.Tensor,
                    pair_batch_size: Optional[int]) -> torch.Tensor:
        if (pair_batch_size is None or pair_batch_size <= 0
                or pair_batch_size * 2 != value.size(0)):
            return value
        return torch.cat([
            value[pair_batch_size:], value[:pair_batch_size]
        ], dim=0)

    def forward(self,
                groups: torch.Tensor,
                gate: torch.Tensor,
                probs: torch.Tensor,
                pair_batch_size: Optional[int]) -> torch.Tensor:
        batch_size, _, num_groups, height, width = groups.shape
        assert num_groups == self.num_groups

        response = groups.mean(dim=1)
        local_context = F.avg_pool2d(
            response, kernel_size=3, stride=1, padding=1)
        detail = (response - local_context).abs()
        detail_scale = detail.square().mean(
            dim=(-2, -1), keepdim=True).add(1e-6).sqrt()
        detail_mask = torch.tanh(detail / detail_scale)

        detail_stats = torch.stack([
            detail.mean(dim=(-2, -1)),
            detail.flatten(2).std(dim=-1, unbiased=False),
        ], dim=-1)
        other_stats = self._pair_other(detail_stats, pair_batch_size)

        coverage = F.normalize(probs.sum(dim=2), p=1, dim=-1)
        other_coverage = self._pair_other(coverage, pair_batch_size)
        coverage_intersection = torch.minimum(
            coverage, other_coverage).sum(dim=-1, keepdim=True)
        coverage_distance = 0.5 * (
            coverage - other_coverage).abs().sum(dim=-1, keepdim=True)
        pair_descriptor = torch.cat([
            detail_stats,
            other_stats,
            detail_stats - other_stats,
            detail_stats * other_stats,
            coverage_intersection,
            coverage_distance,
        ], dim=-1)
        hidden = (self.gain_mlp[0](pair_descriptor) +
                  self.group_embedding.unsqueeze(0))
        gain = torch.tanh(self.gain_mlp[2](self.gain_mlp[1](hidden)))
        gain = gain.view(batch_size, num_groups, 1, 1)

        self.last_detail_gain = gain.detach()
        self.last_detail_mask_mean = detail_mask.detach().mean()
        weighted_detail = gate * detail_mask * gain
        return (groups * weighted_detail.unsqueeze(1)).sum(dim=2)


class PairAlignedCompactDetailEnhancement(nn.Module):
    """Restore compact detail without disturbing confident SE decisions.

    Paired frames share group indices, so the gain is estimated directly from
    same-group detail statistics. Spatial masks remain frame-specific to avoid
    assuming pixel alignment between moving targets.
    """

    def __init__(self,
                 num_groups: int,
                 hidden_dims: int = 16) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.group_embedding = nn.Parameter(torch.zeros(
            num_groups, hidden_dims))
        self.gain_mlp = nn.Sequential(
            nn.Linear(6, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, 1),
        )
        nn.init.normal_(self.group_embedding, std=0.02)
        nn.init.xavier_uniform_(self.gain_mlp[0].weight)
        nn.init.zeros_(self.gain_mlp[0].bias)
        nn.init.zeros_(self.gain_mlp[-1].weight)
        nn.init.zeros_(self.gain_mlp[-1].bias)
        self.last_detail_gain = None
        self.last_compact_mask_mean = None
        self.last_uncertainty_mean = None

    def _pair_descriptor(self, stats: torch.Tensor,
                         pair_batch_size: Optional[int]) -> torch.Tensor:
        batch_size = stats.size(0)
        if (pair_batch_size is None or pair_batch_size <= 0
                or pair_batch_size * 2 != batch_size):
            return torch.cat([
                stats, torch.zeros_like(stats), stats.square()
            ], dim=-1)

        prev = stats[:pair_batch_size]
        curr = stats[pair_batch_size:]
        shared = torch.cat([
            0.5 * (prev + curr),
            (prev - curr).abs(),
            prev * curr,
        ], dim=-1)
        return torch.cat([shared, shared], dim=0)

    def forward(self,
                groups: torch.Tensor,
                gate: torch.Tensor,
                pair_batch_size: Optional[int],
                response: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, num_groups, _, _ = groups.shape
        assert num_groups == self.num_groups

        if response is None:
            response = groups.mean(dim=1)
        else:
            assert response.shape == groups.shape[:1] + groups.shape[2:]
        center_surround = (response - F.avg_pool2d(
            response, kernel_size=3, stride=1, padding=1)).abs()
        local_density = F.avg_pool2d(
            center_surround, kernel_size=5, stride=1, padding=2)
        compact_detail = F.relu(center_surround - local_density)
        detail_scale = compact_detail.square().mean(
            dim=(-2, -1), keepdim=True).add(1e-6).sqrt()
        compact_mask = torch.tanh(compact_detail / detail_scale)

        detail_stats = torch.stack([
            compact_detail.mean(dim=(-2, -1)),
            compact_detail.flatten(2).std(dim=-1, unbiased=False),
        ], dim=-1)
        descriptor = self._pair_descriptor(
            detail_stats, pair_batch_size)
        hidden = (self.gain_mlp[0](descriptor) +
                  self.group_embedding.unsqueeze(0))
        gain = torch.tanh(self.gain_mlp[2](self.gain_mlp[1](hidden)))
        gain = gain.view(batch_size, num_groups, 1, 1)

        # Only ambiguous SE decisions receive a detail correction. Confidently
        # accepted or rejected groups keep the original fusion path.
        uncertainty = 4.0 * gate * (1.0 - gate)
        weighted_detail = uncertainty * compact_mask * gain
        self.last_detail_gain = gain.detach()
        self.last_compact_mask_mean = compact_mask.detach().mean()
        self.last_uncertainty_mean = uncertainty.detach().mean()
        return weighted_detail


class PairAwareLiquidFusion(nn.Module):
    """Generate paired-frame SE residuals from liquid descriptors.

    The module is intentionally not a band-attention block.  It sees only
    compact per-frame group descriptors and predicts a residual for each
    frame's SE gate using prev/curr difference and agreement cues.
    """

    def __init__(self,
                 num_groups: int,
                 num_spectral: int,
                 spectral_kernel: int,
                 hidden_dims: int = 32,
                 init_std: float = 1e-3,
                 zero_init: bool = True) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.num_spectral = num_spectral
        self.spectral_kernel = spectral_kernel
        descriptor_dims = num_spectral + 3
        pair_dims = descriptor_dims * 4
        self.group_embedding = nn.Parameter(torch.zeros(num_groups, hidden_dims))
        self.mlp = nn.Sequential(
            nn.Linear(pair_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, 1),
        )
        self._init_weights(init_std, zero_init)

    def _init_weights(self, init_std: float, zero_init: bool) -> None:
        nn.init.trunc_normal_(self.group_embedding, std=init_std)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
        else:
            nn.init.normal_(self.mlp[-1].weight, std=init_std)
        nn.init.zeros_(self.mlp[-1].bias)

    def _descriptor(self, x_se: torch.Tensor,
                    probs: torch.Tensor) -> torch.Tensor:
        coverage = probs.sum(dim=2)
        coverage = coverage / coverage.sum(dim=-1, keepdim=True).clamp_min(
            1e-6)
        entropy = -(coverage.clamp_min(1e-6) *
                    coverage.clamp_min(1e-6).log()).sum(dim=-1, keepdim=True)
        entropy = entropy / math.log(self.num_spectral)
        peak = coverage.amax(dim=-1, keepdim=True)
        response = x_se.detach().abs().mean(dim=(-2, -1), keepdim=False)
        response = response.unsqueeze(-1)
        return torch.cat([coverage, entropy, peak, response], dim=-1)

    def forward(self, x_se: torch.Tensor, probs: torch.Tensor,
                pair_batch_size: Optional[int]) -> torch.Tensor:
        batch_size, num_groups, height, width = x_se.shape
        if (pair_batch_size is None or pair_batch_size <= 0
                or pair_batch_size * 2 != batch_size):
            return x_se.new_zeros(batch_size, num_groups, height, width)
        assert num_groups == self.num_groups
        desc = self._descriptor(x_se, probs)
        prev_desc = desc[:pair_batch_size]
        curr_desc = desc[pair_batch_size:]

        def _pair_input(src: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [src, other, src - other, src * other], dim=-1)

        prev_pair = _pair_input(prev_desc, curr_desc)
        curr_pair = _pair_input(curr_desc, prev_desc)
        pair_desc = torch.cat([prev_pair, curr_pair], dim=0)
        hidden = self.mlp[0](pair_desc) + self.group_embedding.unsqueeze(0)
        hidden = self.mlp[1](hidden)
        delta = self.mlp[2](hidden).view(batch_size, num_groups, 1, 1)
        return delta.expand(batch_size, num_groups, height, width)


class BandSlotAdaptiveCalibration(nn.Module):
    """Calibrate each sampled physical band for its Conv3D kernel slot."""

    def __init__(self,
                 num_spectral: int,
                 spectral_kernel: int,
                 max_relative_delta: float = 1.0) -> None:
        super().__init__()
        assert num_spectral > 0
        assert spectral_kernel > 0
        assert 0.0 < max_relative_delta <= 1.0
        self.max_relative_delta = float(max_relative_delta)
        self.band_slot_log_scale = nn.Parameter(
            torch.zeros(spectral_kernel, num_spectral))
        self.last_scale = None

    def forward(self, sampled: torch.Tensor,
                probs: torch.Tensor) -> torch.Tensor:
        assert sampled.ndim == 5
        assert probs.ndim == 4
        assert sampled.shape[:3] == probs.shape[:3]
        slot_evidence = torch.einsum(
            'bgks,ks->bgk', probs, self.band_slot_log_scale)
        scale = 1.0 + self.max_relative_delta * torch.tanh(slot_evidence)
        self.last_scale = scale.detach()
        return sampled * scale.unsqueeze(-1).unsqueeze(-1)


class DispersionAwareSpectralEvidence(nn.Module):
    """Mix channel mean and RMS into one evidence map per spectral group."""

    def __init__(self, num_groups: int, eps: float = 1e-6) -> None:
        super().__init__()
        assert num_groups > 0
        assert eps > 0
        self.num_groups = num_groups
        self.eps = float(eps)
        self.evidence_mixer = nn.Conv2d(
            num_groups * 2,
            num_groups,
            kernel_size=1,
            groups=num_groups,
            bias=False)
        self._init_identity()

    def _init_identity(self) -> None:
        nn.init.zeros_(self.evidence_mixer.weight)
        with torch.no_grad():
            self.evidence_mixer.weight[:, 0, 0, 0] = 1.0

    def forward(self, groups: torch.Tensor) -> torch.Tensor:
        assert groups.ndim == 5
        assert groups.size(2) == self.num_groups
        channel_mean = groups.mean(dim=1)
        channel_rms = torch.linalg.vector_norm(
            groups, ord=2, dim=1) / math.sqrt(groups.size(1))
        channel_rms = channel_rms.clamp_min(self.eps)
        statistics = torch.stack(
            [channel_mean, channel_rms], dim=2).flatten(1, 2)
        return self.evidence_mixer(statistics)


class ConsistencyPreservingDispersionEvidence(nn.Module):
    """Add bounded dispersion evidence without replacing mean evidence.

    The local variant retains spatially sparse evidence. The pair-global
    variant shares one group-wise correction across both frames, so the
    branch cannot alter their relative spatial ordering.
    """

    def __init__(self,
                 num_groups: int,
                 mode: str = 'local',
                 max_logit_delta: float = 0.5,
                 center_groups: bool = False,
                 preserve_detection_tangent: bool = False,
                 preserve_sparse_detection_evidence: bool = False,
                 eps: float = 1e-6) -> None:
        super().__init__()
        assert num_groups > 0
        assert mode in ('local', 'pair_global')
        assert max_logit_delta > 0
        assert eps > 0
        self.num_groups = num_groups
        self.mode = mode
        self.max_logit_delta = float(max_logit_delta)
        self.center_groups = bool(center_groups)
        self.preserve_detection_tangent = bool(
            preserve_detection_tangent)
        self.preserve_sparse_detection_evidence = bool(
            preserve_sparse_detection_evidence)
        assert sum((
            self.center_groups,
            self.preserve_detection_tangent,
            self.preserve_sparse_detection_evidence,
        )) <= 1, (
            'group centering, detection-tangent preservation, and sparse '
            'detection evidence preservation are alternative residual '
            'constraints')
        if (self.preserve_detection_tangent
                or self.preserve_sparse_detection_evidence):
            assert self.mode == 'pair_global', (
                'detection preservation requires pair_global mode')
        self.eps = float(eps)
        self.logit_gain = nn.Parameter(torch.zeros(num_groups))
        self.last_normalized_dispersion = None
        self.last_detection_importance = None
        self.last_sparse_detection_reserve = None
        self.last_delta = None

    def forward(self, groups: torch.Tensor,
                pair_batch_size: Optional[int],
                gate_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert groups.ndim == 5
        assert groups.size(2) == self.num_groups
        channel_mean = groups.mean(dim=1)
        second_moment = groups.square().mean(dim=1)
        variance = (second_moment - channel_mean.square()).clamp_min(self.eps)
        normalized = (
            variance.sqrt() /
            second_moment.clamp_min(self.eps).sqrt()).clamp_max(1)

        sparse_detection_reserve = None
        if self.preserve_sparse_detection_evidence:
            spatial_mean = normalized.mean(dim=(-2, -1))
            spatial_rms = torch.linalg.vector_norm(
                normalized, ord=2, dim=(-2, -1)) / math.sqrt(
                    normalized.size(-2) * normalized.size(-1))
            frame_reserve = (spatial_rms - spatial_mean).clamp_(0, 1)
            assert pair_batch_size is not None and pair_batch_size > 0
            sparse_detection_reserve = 0.5 * (
                frame_reserve[:pair_batch_size] +
                frame_reserve[pair_batch_size:])
            sparse_detection_reserve = torch.cat(
                [sparse_detection_reserve, sparse_detection_reserve],
                dim=0).detach()

        if self.mode == 'pair_global':
            assert pair_batch_size is not None and pair_batch_size > 0
            assert groups.size(0) == pair_batch_size * 2
            pooled = normalized.mean(dim=(-2, -1))
            shared = 0.5 * (
                pooled[:pair_batch_size] + pooled[pair_batch_size:])
            normalized = torch.cat([shared, shared], dim=0)
            normalized = normalized.unsqueeze(-1).unsqueeze(-1)

        gain = self.max_logit_delta * torch.tanh(self.logit_gain)
        gain = gain.to(dtype=normalized.dtype)
        delta = normalized * gain.view(1, -1, 1, 1)
        if self.center_groups:
            delta = delta - delta.mean(dim=1, keepdim=True)
        if self.preserve_sparse_detection_evidence:
            assert sparse_detection_reserve is not None
            negative_scale = (
                1 - sparse_detection_reserve).to(dtype=delta.dtype)
            negative_scale = negative_scale.unsqueeze(-1).unsqueeze(-1)
            delta = torch.where(delta < 0, delta * negative_scale, delta)
            self.last_sparse_detection_reserve = (
                sparse_detection_reserve.detach())
        else:
            self.last_sparse_detection_reserve = None
        if self.preserve_detection_tangent:
            assert gate_logits is not None
            assert gate_logits.shape == (
                groups.size(0), self.num_groups,
                groups.size(-2), groups.size(-1))
            # Reuse the moment already needed by the dispersion descriptor;
            # recomputing it from [B,C,G,H,W] measurably slows the full stem.
            group_energy = second_moment.detach().mean(
                dim=(-2, -1)).add(self.eps).sqrt()
            pooled_logits = gate_logits.detach().mean(dim=(-2, -1))
            pooled_gate = pooled_logits.sigmoid()
            gate_sensitivity = pooled_gate * (1 - pooled_gate)
            importance = group_energy * gate_sensitivity
            shared_importance = 0.5 * (
                importance[:pair_batch_size] +
                importance[pair_batch_size:])
            importance = torch.cat(
                [shared_importance, shared_importance], dim=0)
            delta_vector = delta.flatten(1)
            projection_scale = (
                (delta_vector * importance).sum(dim=1, keepdim=True) /
                importance.square().sum(
                    dim=1, keepdim=True).clamp_min(self.eps))
            delta = (
                delta_vector - projection_scale * importance
            ).unsqueeze(-1).unsqueeze(-1)
            self.last_detection_importance = importance.detach()
        else:
            self.last_detection_importance = None
        self.last_normalized_dispersion = normalized.detach()
        self.last_delta = delta.detach()
        return delta


class SpectralCoordinatePairDispersion(nn.Module):
    """Inject pair dispersion in physical spectral coordinates.

    Group slots may select different bands in paired frames. This module first
    projects group dispersion back to physical bands, forms one pair-common
    spectral descriptor, and then projects it into each frame's own groups.
    Evidence and coverage are read-only so this branch cannot steer the
    sampler or Conv3D features to manufacture an easier correction.
    """

    def __init__(self,
                 num_groups: int,
                 num_spectral: int,
                 max_logit_delta: float = 0.25,
                 eps: float = 1e-6) -> None:
        super().__init__()
        assert num_groups > 0
        assert num_spectral > 0
        assert max_logit_delta > 0
        assert eps > 0
        self.num_groups = num_groups
        self.num_spectral = num_spectral
        self.max_logit_delta = float(max_logit_delta)
        self.eps = float(eps)
        self.spectral_logit_gain = nn.Parameter(torch.zeros(num_spectral))
        self.last_common_spectral_evidence = None
        self.last_delta = None

    def _group_dispersion(self, groups: torch.Tensor) -> torch.Tensor:
        channel_mean = groups.mean(dim=1)
        second_moment = groups.square().mean(dim=1)
        variance = (second_moment - channel_mean.square()).clamp_min(self.eps)
        normalized = (
            variance.sqrt() /
            second_moment.clamp_min(self.eps).sqrt()).clamp_max(1)
        return normalized.mean(dim=(-2, -1))

    def forward(self, groups: torch.Tensor, probs: torch.Tensor,
                pair_batch_size: Optional[int]) -> torch.Tensor:
        assert groups.ndim == 5
        assert groups.size(2) == self.num_groups
        assert probs.ndim == 4
        assert probs.shape[:2] == groups.shape[:1] + (self.num_groups, )
        assert probs.size(-1) == self.num_spectral
        batch_size = groups.size(0)
        if (pair_batch_size is None or pair_batch_size <= 0
                or pair_batch_size * 2 != batch_size):
            return groups.new_zeros(batch_size, self.num_groups, 1, 1)

        # Keep the auxiliary evidence branch from changing route optimization
        # or the Conv3D response path through its descriptor construction.
        dispersion = self._group_dispersion(groups).detach()
        coverage = probs.detach().sum(dim=2)
        coverage = coverage / coverage.sum(dim=-1, keepdim=True).clamp_min(
            self.eps)

        def _to_spectral(frame_coverage: torch.Tensor,
                         frame_dispersion: torch.Tensor) -> torch.Tensor:
            numerator = torch.einsum(
                'bgs,bg->bs', frame_coverage, frame_dispersion)
            denominator = frame_coverage.sum(dim=1).clamp_min(self.eps)
            return numerator / denominator

        prev_coverage = coverage[:pair_batch_size]
        curr_coverage = coverage[pair_batch_size:]
        prev_spectral = _to_spectral(
            prev_coverage, dispersion[:pair_batch_size])
        curr_spectral = _to_spectral(
            curr_coverage, dispersion[pair_batch_size:])
        common_spectral = 0.5 * (prev_spectral + curr_spectral)

        gain = self.max_logit_delta * torch.tanh(
            self.spectral_logit_gain)
        weighted_spectral = common_spectral * gain.to(
            dtype=common_spectral.dtype)
        prev_delta = torch.einsum(
            'bgs,bs->bg', prev_coverage, weighted_spectral)
        curr_delta = torch.einsum(
            'bgs,bs->bg', curr_coverage, weighted_spectral)
        delta = torch.cat([prev_delta, curr_delta], dim=0)
        # Preserve the per-frame average SE logit to first order.
        delta = delta - delta.mean(dim=1, keepdim=True)
        delta = delta.unsqueeze(-1).unsqueeze(-1)
        self.last_common_spectral_evidence = common_spectral.detach()
        self.last_delta = delta.detach()
        return delta


class PairEvidenceConsensusGate(nn.Module):
    """Contract paired SE gates only where their evidence agrees.

    The correction is antisymmetric, so the pair mean gate is preserved
    exactly. Detached agreement weights prevent the sampler or Conv3D
    responses from gaming the consensus strength.
    """

    def __init__(self,
                 num_groups: int,
                 max_strength: float = 1.0,
                 init_logit: float = -4.0,
                 eps: float = 1e-6) -> None:
        super().__init__()
        assert num_groups > 0
        assert 0 < max_strength <= 1
        assert eps > 0
        self.num_groups = num_groups
        self.max_strength = float(max_strength)
        self.eps = float(eps)
        self.strength_logit = nn.Parameter(
            torch.full((num_groups, ), float(init_logit)))
        self.last_strength = None
        self.last_route_agreement = None
        self.last_evidence_agreement = None
        self.last_correction = None

    def forward(self, gate: torch.Tensor, evidence: torch.Tensor,
                probs: torch.Tensor,
                pair_batch_size: Optional[int]) -> torch.Tensor:
        assert gate.ndim == evidence.ndim == 4
        assert gate.shape == evidence.shape
        assert gate.size(1) == self.num_groups
        assert probs.ndim == 4
        assert probs.shape[:2] == gate.shape[:2]
        batch_size = gate.size(0)
        if (pair_batch_size is None or pair_batch_size <= 0
                or pair_batch_size * 2 != batch_size):
            self.last_strength = None
            self.last_route_agreement = None
            self.last_evidence_agreement = None
            self.last_correction = None
            return gate

        prev_gate = gate[:pair_batch_size]
        curr_gate = gate[pair_batch_size:]
        prev_evidence = evidence[:pair_batch_size]
        curr_evidence = evidence[pair_batch_size:]

        coverage = probs.sum(dim=2)
        coverage = coverage / coverage.sum(
            dim=-1, keepdim=True).clamp_min(self.eps)
        prev_coverage = coverage[:pair_batch_size]
        curr_coverage = coverage[pair_batch_size:]
        route_agreement = (
            prev_coverage.clamp_min(0).sqrt()
            * curr_coverage.clamp_min(0).sqrt()).sum(dim=-1)
        route_agreement = route_agreement.clamp(0, 1).detach()

        relative_gap = (
            (prev_evidence - curr_evidence).abs()
            / (prev_evidence.abs() + curr_evidence.abs()).clamp_min(self.eps))
        evidence_agreement = (1.0 - relative_gap).clamp(0, 1).detach()

        strength = self.max_strength * self.strength_logit.sigmoid()
        strength = strength.to(dtype=gate.dtype)
        correction = 0.5 * (prev_gate - curr_gate)
        correction = correction * evidence_agreement
        correction = correction * route_agreement.unsqueeze(-1).unsqueeze(-1)
        correction = correction * strength.view(1, -1, 1, 1)
        output = torch.cat(
            [prev_gate - correction, curr_gate + correction], dim=0)

        self.last_strength = strength.detach()
        self.last_route_agreement = route_agreement.detach()
        self.last_evidence_agreement = evidence_agreement.detach()
        self.last_correction = correction.detach()
        return output


@MODELS.register_module()
class MultispecStemConv3dSE(nn.Module):
    """Replace deep-stem first 3x3 Conv2d with 3D conv + pixel-wise SE fusion.

    Input shape ``[B, num_spectral, H, W]`` is treated as
    ``[B, 1, num_spectral, H, W]`` for spectral 3D convolution.

    The 3D kernel is ``(spectral, H, W) = (3, 3, 3)``, **not** ``(3, 7, 7)``.
    Spatial size/stride/padding follow ResNetV1d ``stem.0`` (k=3, s=2, p=1) so
    pretrained ``stem.0.weight`` with shape ``(out, 3, 3, 3)`` can be mapped
    to ``conv3d.weight`` with shape ``(out, 1, 3, 3, 3)``. BatchNorm/ReLU
    after this module remain in the ResNet deep stem.

    Args:
        out_channels (int): Output channels, typically ``stem_channels // 2``.
        num_spectral (int): Number of spectral input bands. Defaults to 8.
        spectral_kernel (int): Spectral-axis kernel, fixed to 3 for pretrain.
        spatial_kernel (int): Spatial kernel, must match stem.0 (3).
        spatial_stride (int): Spatial stride, must match stem.0 (2).
        reduction (int): SE bottleneck ratio. Defaults to 4.
        liquid_sampler (dict | None): Optional Liquid Spectral Sampling config.
    """

    def __init__(self,
                 out_channels: int,
                 num_spectral: int = 8,
                 spectral_kernel: int = STEM_SPECTRAL_KERNEL,
                 spatial_kernel: int = STEM_SPATIAL_KERNEL,
                 spatial_stride: int = STEM_SPATIAL_STRIDE,
                 reduction: int = 4,
                 liquid_sampler: Optional[dict] = None) -> None:
        super().__init__()
        assert num_spectral > 1, f'num_spectral must be > 1, got {num_spectral}'
        assert spatial_kernel == STEM_SPATIAL_KERNEL, (
            f'spatial_kernel must be {STEM_SPATIAL_KERNEL} to match ResNetV1d '
            f'stem.0, got {spatial_kernel}')
        assert spatial_stride == STEM_SPATIAL_STRIDE, (
            f'spatial_stride must be {STEM_SPATIAL_STRIDE} to match ResNetV1d '
            f'stem.0, got {spatial_stride}')
        assert spectral_kernel == STEM_SPECTRAL_KERNEL, (
            f'spectral_kernel must be {STEM_SPECTRAL_KERNEL} for RGB pretrain '
            f'mapping, got {spectral_kernel}')

        spectral_padding = spectral_kernel // 2
        spatial_padding = STEM_SPATIAL_PADDING

        self.num_spectral = num_spectral
        self.spectral_kernel = spectral_kernel
        self.spectral_padding = spectral_padding
        self.spatial_padding = spatial_padding
        self.use_liquid_sampler = liquid_sampler is not None
        self.pair_batch_size = None
        self.conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=(spectral_kernel, spatial_kernel, spatial_kernel),
            stride=(1, spatial_stride, spatial_stride),
            padding=(spectral_padding, spatial_padding, spatial_padding),
            bias=False)

        if self.use_liquid_sampler:
            sampler_cfg = dict(liquid_sampler)
            band_slot_cfg = sampler_cfg.pop('band_slot_calibration', None)
            dispersion_evidence_cfg = sampler_cfg.pop(
                'dispersion_aware_spectral_evidence', None)
            consistency_evidence_cfg = sampler_cfg.pop(
                'consistency_preserving_dispersion_evidence', None)
            spectral_coordinate_dispersion_cfg = sampler_cfg.pop(
                'spectral_coordinate_pair_dispersion', None)
            pair_consensus_gate_cfg = sampler_cfg.pop(
                'pair_evidence_consensus_gate', None)
            fusion_cfg = sampler_cfg.pop('liquid_aware_fusion', None)
            pair_fusion_cfg = sampler_cfg.pop('pair_aware_liquid_fusion',
                                              None)
            group_modulator_cfg = sampler_cfg.pop('liquid_group_modulator',
                                                  None)
            detail_preservation_cfg = sampler_cfg.pop(
                'pair_consistent_detail_preservation', None)
            compact_detail_cfg = sampler_cfg.pop(
                'pair_aligned_compact_detail_enhancement', None)
            assert not (detail_preservation_cfg is not None
                        and compact_detail_cfg is not None), (
                            'pair_consistent_detail_preservation and '
                            'pair_aligned_compact_detail_enhancement are '
                            'mutually exclusive')
            sampler_cfg.setdefault('num_spectral', num_spectral)
            sampler_cfg.setdefault('spectral_kernel', spectral_kernel)
            self.liquid_sampler = LiquidSpectralSampler(**sampler_cfg)
            temporal_output_size = self.liquid_sampler.num_groups
        else:
            band_slot_cfg = None
            dispersion_evidence_cfg = None
            consistency_evidence_cfg = None
            spectral_coordinate_dispersion_cfg = None
            pair_consensus_gate_cfg = None
            fusion_cfg = None
            pair_fusion_cfg = None
            group_modulator_cfg = None
            detail_preservation_cfg = None
            compact_detail_cfg = None
            self.liquid_sampler = None
            temporal_output_size = calc_temporal_output_size(
                num_spectral, spectral_padding, spectral_kernel, 1)
        assert temporal_output_size // reduction >= 1, (
            f'SE bottleneck too narrow: temporal={temporal_output_size}, '
            f'reduction={reduction}')

        if band_slot_cfg is True:
            band_slot_cfg = {}
        if band_slot_cfg is not None:
            band_slot_cfg = dict(band_slot_cfg)
            band_slot_cfg.setdefault('num_spectral', num_spectral)
            band_slot_cfg.setdefault('spectral_kernel', spectral_kernel)
            self.band_slot_calibration = BandSlotAdaptiveCalibration(
                **band_slot_cfg)
        else:
            self.band_slot_calibration = None
        if dispersion_evidence_cfg is True:
            dispersion_evidence_cfg = {}
        if dispersion_evidence_cfg is not None:
            dispersion_evidence_cfg = dict(dispersion_evidence_cfg)
            dispersion_evidence_cfg.setdefault(
                'num_groups', temporal_output_size)
            self.dispersion_aware_spectral_evidence = (
                DispersionAwareSpectralEvidence(**dispersion_evidence_cfg))
        else:
            self.dispersion_aware_spectral_evidence = None
        assert not (consistency_evidence_cfg is not None
                    and spectral_coordinate_dispersion_cfg is not None), (
                        'SE-logit dispersion residual branches are mutually '
                        'exclusive')
        if consistency_evidence_cfg is True:
            consistency_evidence_cfg = {}
        if consistency_evidence_cfg is not None:
            consistency_evidence_cfg = dict(consistency_evidence_cfg)
            consistency_evidence_cfg.setdefault(
                'num_groups', temporal_output_size)
            self.consistency_preserving_dispersion_evidence = (
                ConsistencyPreservingDispersionEvidence(
                    **consistency_evidence_cfg))
        else:
            self.consistency_preserving_dispersion_evidence = None
        if spectral_coordinate_dispersion_cfg is True:
            spectral_coordinate_dispersion_cfg = {}
        if spectral_coordinate_dispersion_cfg is not None:
            spectral_coordinate_dispersion_cfg = dict(
                spectral_coordinate_dispersion_cfg)
            spectral_coordinate_dispersion_cfg.setdefault(
                'num_groups', temporal_output_size)
            spectral_coordinate_dispersion_cfg.setdefault(
                'num_spectral', num_spectral)
            self.spectral_coordinate_pair_dispersion = (
                SpectralCoordinatePairDispersion(
                    **spectral_coordinate_dispersion_cfg))
        else:
            self.spectral_coordinate_pair_dispersion = None
        if pair_consensus_gate_cfg is True:
            pair_consensus_gate_cfg = {}
        if pair_consensus_gate_cfg is not None:
            pair_consensus_gate_cfg = dict(pair_consensus_gate_cfg)
            pair_consensus_gate_cfg.setdefault(
                'num_groups', temporal_output_size)
            self.pair_evidence_consensus_gate = PairEvidenceConsensusGate(
                **pair_consensus_gate_cfg)
        else:
            self.pair_evidence_consensus_gate = None

        self.se_conv1 = nn.Conv2d(
            temporal_output_size,
            temporal_output_size // reduction,
            kernel_size=3,
            padding=1,
            bias=True)
        self.se_conv2 = nn.Conv2d(
            temporal_output_size // reduction,
            temporal_output_size,
            kernel_size=3,
            padding=1,
            bias=True)
        self.num_bands = temporal_output_size
        if fusion_cfg is True:
            fusion_cfg = {}
        output_residual_cfg = None
        if fusion_cfg is not None:
            fusion_cfg = dict(fusion_cfg)
            output_residual_cfg = fusion_cfg.pop('output_residual', None)
            quality_conservation_cfg = fusion_cfg.pop(
                'quality_conservation', None)
            fusion_cfg.setdefault('num_groups', temporal_output_size)
            fusion_cfg.setdefault('num_spectral', num_spectral)
            fusion_cfg.setdefault('spectral_kernel', spectral_kernel)
            fusion_cfg.setdefault(
                'use_sparse_evidence',
                self.liquid_sampler.sparse_spectral_evidence is not None)
            self.liquid_aware_fusion = LiquidAwareFusion(**fusion_cfg)
        else:
            quality_conservation_cfg = None
            self.liquid_aware_fusion = None
        if pair_fusion_cfg is True:
            pair_fusion_cfg = {}
        if pair_fusion_cfg is not None:
            pair_fusion_cfg = dict(pair_fusion_cfg)
            pair_fusion_cfg.setdefault('num_groups', temporal_output_size)
            pair_fusion_cfg.setdefault('num_spectral', num_spectral)
            pair_fusion_cfg.setdefault('spectral_kernel', spectral_kernel)
            self.pair_aware_liquid_fusion = PairAwareLiquidFusion(
                **pair_fusion_cfg)
        else:
            self.pair_aware_liquid_fusion = None
        if group_modulator_cfg is True:
            group_modulator_cfg = {}
        if group_modulator_cfg is not None:
            group_modulator_cfg = dict(group_modulator_cfg)
            group_modulator_cfg.setdefault('num_groups', temporal_output_size)
            group_modulator_cfg.setdefault('num_spectral', num_spectral)
            group_modulator_cfg.setdefault('spectral_kernel', spectral_kernel)
            self.liquid_group_modulator = LiquidGroupModulator(
                **group_modulator_cfg)
        else:
            self.liquid_group_modulator = None
        if detail_preservation_cfg is True:
            detail_preservation_cfg = {}
        if detail_preservation_cfg is not None:
            detail_preservation_cfg = dict(detail_preservation_cfg)
            detail_preservation_cfg.setdefault(
                'num_groups', temporal_output_size)
            detail_preservation_cfg.setdefault('num_spectral', num_spectral)
            self.pair_consistent_detail_preservation = (
                PairConsistentDetailPreservation(**detail_preservation_cfg))
        else:
            self.pair_consistent_detail_preservation = None
        if compact_detail_cfg is True:
            compact_detail_cfg = {}
        if compact_detail_cfg is not None:
            compact_detail_cfg = dict(compact_detail_cfg)
            compact_detail_cfg.setdefault(
                'num_groups', temporal_output_size)
            self.pair_aligned_compact_detail_enhancement = (
                PairAlignedCompactDetailEnhancement(**compact_detail_cfg))
        else:
            self.pair_aligned_compact_detail_enhancement = None
        if output_residual_cfg is True:
            output_residual_cfg = {}
        if output_residual_cfg is not None:
            init_value = float(output_residual_cfg.get('init_value', 0.05))
            self.liquid_output_residual_scale = nn.Parameter(
                torch.tensor(init_value, dtype=torch.float32))
        else:
            self.liquid_output_residual_scale = None
        if quality_conservation_cfg is True:
            quality_conservation_cfg = {}
        if quality_conservation_cfg is not None:
            self.fusion_quality_conservation = FusionQualityConservation(
                **dict(quality_conservation_cfg))
        else:
            self.fusion_quality_conservation = None
        self.last_liquid_groups = None
        self.last_liquid_probs = None
        self.last_liquid_context_probs = None
        self.last_liquid_indices = None
        self.last_liquid_aware_delta = None
        self.last_pair_aware_liquid_delta = None
        self._init_se_weights()

    def set_pair_batch_size(self, pair_batch_size: Optional[int]) -> None:
        self.pair_batch_size = pair_batch_size

    def _init_se_weights(self) -> None:
        """Init SE so gate starts uniform: each band weight is ``1 / T``.

        With ``se_conv1`` output zeroed, ``se_conv2`` bias is set to
        ``logit(1/T)``, hence ``sigmoid(...) == 1/T`` and spectral fusion
        begins as an equal-weight average across bands.
        """
        nn.init.zeros_(self.se_conv1.weight)
        nn.init.zeros_(self.se_conv1.bias)
        nn.init.zeros_(self.se_conv2.weight)
        uniform_bias = uniform_gate_logit(self.num_bands)
        nn.init.constant_(self.se_conv2.bias, uniform_bias)

    def _forward_fixed(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        return self.conv3d(x)

    def _forward_liquid(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = x.squeeze(1)
        sampled, probs = self.liquid_sampler(
            x, self.pair_batch_size, preview_weight=self.conv3d.weight)
        if self.band_slot_calibration is not None:
            sampled = self.band_slot_calibration(sampled, probs)
        batch_size, num_groups, _, height, width = sampled.shape
        sampled = sampled.reshape(
            batch_size, 1, num_groups * self.spectral_kernel, height, width)
        groups = F.conv3d(
            sampled,
            self.conv3d.weight,
            self.conv3d.bias,
            stride=(self.spectral_kernel, self.conv3d.stride[1],
                    self.conv3d.stride[2]),
            padding=(0, self.spatial_padding, self.spatial_padding),
            dilation=self.conv3d.dilation,
            groups=self.conv3d.groups)
        self.last_liquid_groups = groups
        self.last_liquid_probs = probs
        self.last_liquid_context_probs = self.liquid_sampler.last_context_probs
        self.last_liquid_indices = self.liquid_sampler.last_hard_indices
        return groups

    def forward(self,
                x: torch.Tensor,
                return_sampling: bool = False):
        if self.use_liquid_sampler:
            x = self._forward_liquid(x)
        else:
            x = self._forward_fixed(x)

        if self.liquid_group_modulator is not None:
            assert self.last_liquid_context_probs is not None
            x = self.liquid_group_modulator(
                x, self.last_liquid_context_probs)

        if self.dispersion_aware_spectral_evidence is not None:
            x_se = self.dispersion_aware_spectral_evidence(x)
        else:
            x_se = x.mean(dim=1)
        gate_logits = self.se_conv2(F.relu(self.se_conv1(x_se)))
        if self.consistency_preserving_dispersion_evidence is not None:
            gate_logits = gate_logits + (
                self.consistency_preserving_dispersion_evidence(
                    x, self.pair_batch_size, gate_logits=gate_logits))
        if self.spectral_coordinate_pair_dispersion is not None:
            assert self.last_liquid_probs is not None
            gate_logits = gate_logits + (
                self.spectral_coordinate_pair_dispersion(
                    x, self.last_liquid_probs, self.pair_batch_size))
        if self.liquid_aware_fusion is not None:
            assert self.last_liquid_context_probs is not None
            self.last_liquid_aware_delta = self.liquid_aware_fusion(
                x_se,
                self.last_liquid_context_probs,
                self.pair_batch_size,
                self.liquid_sampler.last_pair_band_context,
                self.liquid_sampler.last_sparse_band_evidence,
                self.liquid_sampler.last_sparse_group_map)
            if self.fusion_quality_conservation is not None:
                self.last_liquid_aware_delta = (
                    self.fusion_quality_conservation(
                        gate_logits, self.last_liquid_aware_delta, x_se))
            gate_logits = gate_logits + self.last_liquid_aware_delta
        else:
            self.last_liquid_aware_delta = None
        if self.pair_aware_liquid_fusion is not None:
            assert self.last_liquid_probs is not None
            self.last_pair_aware_liquid_delta = self.pair_aware_liquid_fusion(
                x_se, self.last_liquid_probs, self.pair_batch_size)
            gate_logits = gate_logits + self.last_pair_aware_liquid_delta
        else:
            self.last_pair_aware_liquid_delta = None
        gate = torch.sigmoid(gate_logits)
        if self.pair_evidence_consensus_gate is not None:
            assert self.last_liquid_probs is not None
            gate = self.pair_evidence_consensus_gate(
                gate, x_se, self.last_liquid_probs, self.pair_batch_size)
        groups_for_detail = x
        if self.pair_aligned_compact_detail_enhancement is not None:
            detail_gate = self.pair_aligned_compact_detail_enhancement(
                groups_for_detail, gate, self.pair_batch_size, x_se)
            fusion_gate = gate + detail_gate
            out = (groups_for_detail * fusion_gate.unsqueeze(1)).sum(dim=2)
            # Keep return_sampling unchanged without materializing this second
            # large product during normal backbone execution.
            x = (groups_for_detail * gate.unsqueeze(1)
                 if return_sampling
                 or self.liquid_output_residual_scale is not None else None)
        else:
            x = groups_for_detail * gate.unsqueeze(1)
            out = x.sum(dim=2)
        if self.pair_consistent_detail_preservation is not None:
            assert self.last_liquid_context_probs is not None
            out = out + self.pair_consistent_detail_preservation(
                groups_for_detail, gate, self.last_liquid_context_probs,
                self.pair_batch_size)
        if (self.liquid_output_residual_scale is not None
                and self.last_liquid_aware_delta is not None):
            assert x is not None
            residual_gate = torch.tanh(
                self.last_liquid_aware_delta).unsqueeze(1)
            out = out + self.liquid_output_residual_scale * (
                x * residual_gate).sum(dim=2)
        if return_sampling:
            assert x is not None
            return out, x, self.last_liquid_probs
        return out
