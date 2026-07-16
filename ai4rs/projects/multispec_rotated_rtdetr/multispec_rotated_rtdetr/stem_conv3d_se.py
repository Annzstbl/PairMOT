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
                 soft_group_set_transport: Optional[dict] = None,
                 pair_sampler_router: Optional[dict] = None,
                 pair_band_context: Optional[dict] = None) -> None:
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
        assert self.set_transport_num_iters > 0
        assert self.set_transport_temperature > 0
        assert 0.0 <= self.set_transport_strength <= 1.0
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
        self.last_set_assignment = None
        self.last_set_max_load = None

        self.desc_proj = nn.Linear(3, embed_dims)
        self.band_embedding = nn.Parameter(torch.zeros(num_spectral, embed_dims))
        self.w1 = nn.Linear(embed_dims * 2, embed_dims)
        self.w2 = nn.Linear(embed_dims * 2, embed_dims)
        self.head = nn.Linear(
            embed_dims, self.num_groups * spectral_kernel * num_spectral)
        if pair_sampler_router is True:
            pair_sampler_router = {}
        if pair_sampler_router is not None:
            router_cfg = dict(pair_sampler_router)
            router_cfg.setdefault('embed_dims', embed_dims)
            router_cfg.setdefault(
                'output_dims',
                self.num_groups * spectral_kernel * num_spectral)
            self.pair_sampler_router = PairCoupledSamplerRouter(**router_cfg)
        else:
            self.pair_sampler_router = None
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
        self.last_pair_band_context = None
        self.last_pair_band_logits = None
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
            return raw_probs
        projected = self._project_soft_group_sets(raw_probs)
        strength = min(max(self.set_transport_strength, 0.0), 1.0)
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
            return hard_probs - soft_probs.detach() + soft_probs

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
            return probs

        probs = F.softmax(logits / self.tau, dim=-1)
        indices = self._dedup_hard_indices(logits)
        self.last_hard_indices = indices
        hard_probs = torch.zeros_like(probs).scatter_(
            -1, indices.unsqueeze(-1), 1.0)
        return hard_probs - probs.detach() + probs

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        self.last_hard_indices = self._dedup_hard_indices(logits)
        if self.training or not self.deterministic_eval:
            if self.hard:
                return self._sample_hard_unique(logits)
            probs = F.gumbel_softmax(
                logits, tau=self.tau, hard=self.hard, dim=-1)
            return self._apply_soft_group_set_transport(probs)

        probs = F.softmax(logits / self.tau, dim=-1)
        if not self.eval_hard:
            return self._apply_soft_group_set_transport(probs)
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

    def forward(self,
                x: torch.Tensor,
                pair_batch_size: Optional[int] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim == 4, f'Expected [B, S, H, W], got {tuple(x.shape)}'
        assert x.size(1) == self.num_spectral, (
            f'Expected {self.num_spectral} spectral bands, got {x.size(1)}')

        mean = x.mean(dim=(-2, -1))
        std = x.flatten(2).std(dim=-1)
        maxv = x.amax(dim=(-2, -1))
        desc = torch.stack([mean, std, maxv], dim=-1)
        desc = self.desc_proj(desc) + self.band_embedding.unsqueeze(0)
        if self.band_attn is not None:
            attn_input = self.band_norm1(desc)
            desc = desc + self.band_attn(attn_input, attn_input, attn_input)[0]
            desc = desc + self.band_ffn(self.band_norm2(desc))
        if self.pair_band_context is not None:
            (desc, self.last_pair_band_context,
             self.last_pair_band_logits) = self.pair_band_context(
                 desc, pair_batch_size)
        else:
            self.last_pair_band_context = None
            self.last_pair_band_logits = None

        hidden = desc.new_zeros(desc.size(0), self.embed_dims)
        for band_idx in range(self.num_spectral):
            cell_input = torch.cat([desc[:, band_idx], hidden], dim=-1)
            h_hat = torch.tanh(self.w1(cell_input))
            alpha = torch.sigmoid(self.w2(cell_input))
            hidden = alpha * hidden + (1 - alpha) * h_hat

        logits = self.head(hidden).view(
            x.size(0), self.num_groups, self.spectral_kernel,
            self.num_spectral)
        if self.pair_sampler_router is not None:
            pair_logits = self.pair_sampler_router(hidden, pair_batch_size)
            logits = logits + pair_logits.view_as(logits)
        if self.last_pair_band_logits is not None:
            logits = logits + self.last_pair_band_logits.view_as(logits)
        probs = self._sample(logits)
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
                 pair_band_context_fusion: Optional[dict] = None,
                 pair_change_gate: Optional[dict] = None) -> None:
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

        pattern_dims = spectral_kernel * num_spectral
        self.pattern_proj = nn.Linear(pattern_dims, embed_dims)
        self.response_proj = nn.Linear(2, embed_dims)
        if use_overlap_context:
            self.overlap_proj = nn.Linear(embed_dims, embed_dims)
        else:
            self.overlap_proj = None
        self.group_embedding = nn.Parameter(torch.zeros(num_groups, embed_dims))
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
        if pair_transport is not None:
            pair_transport_cfg = dict(pair_transport)
            pair_transport_cfg.setdefault('embed_dims', embed_dims)
            self.pair_transport = PairTransportTokenCoupling(
                **pair_transport_cfg)
        else:
            self.pair_transport = None
        self.last_pair_transport = None
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
                pair_band_context: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        batch_size, num_groups, height, width = x_se.shape
        assert num_groups == self.num_groups
        pattern = probs.reshape(batch_size, num_groups, -1)
        spatial_mean = x_se.mean(dim=(-2, -1))
        spatial_std = x_se.flatten(2).std(dim=-1)
        response = torch.stack([spatial_mean, spatial_std], dim=-1)

        token = (self.pattern_proj(pattern) + self.response_proj(response) +
                 self.group_embedding.unsqueeze(0))
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
            fusion_cfg = sampler_cfg.pop('liquid_aware_fusion', None)
            pair_fusion_cfg = sampler_cfg.pop('pair_aware_liquid_fusion',
                                              None)
            group_modulator_cfg = sampler_cfg.pop('liquid_group_modulator',
                                                  None)
            sampler_cfg.setdefault('num_spectral', num_spectral)
            sampler_cfg.setdefault('spectral_kernel', spectral_kernel)
            self.liquid_sampler = LiquidSpectralSampler(**sampler_cfg)
            temporal_output_size = self.liquid_sampler.num_groups
        else:
            fusion_cfg = None
            pair_fusion_cfg = None
            group_modulator_cfg = None
            self.liquid_sampler = None
            temporal_output_size = calc_temporal_output_size(
                num_spectral, spectral_padding, spectral_kernel, 1)
        assert temporal_output_size // reduction >= 1, (
            f'SE bottleneck too narrow: temporal={temporal_output_size}, '
            f'reduction={reduction}')

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
            fusion_cfg.setdefault('num_groups', temporal_output_size)
            fusion_cfg.setdefault('num_spectral', num_spectral)
            fusion_cfg.setdefault('spectral_kernel', spectral_kernel)
            self.liquid_aware_fusion = LiquidAwareFusion(**fusion_cfg)
        else:
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
        if output_residual_cfg is True:
            output_residual_cfg = {}
        if output_residual_cfg is not None:
            init_value = float(output_residual_cfg.get('init_value', 0.05))
            self.liquid_output_residual_scale = nn.Parameter(
                torch.tensor(init_value, dtype=torch.float32))
        else:
            self.liquid_output_residual_scale = None
        self.last_liquid_groups = None
        self.last_liquid_probs = None
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
        sampled, probs = self.liquid_sampler(x, self.pair_batch_size)
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
            assert self.last_liquid_probs is not None
            x = self.liquid_group_modulator(x, self.last_liquid_probs)

        x_se = x.mean(dim=1)
        gate_logits = self.se_conv2(F.relu(self.se_conv1(x_se)))
        if self.liquid_aware_fusion is not None:
            assert self.last_liquid_probs is not None
            self.last_liquid_aware_delta = self.liquid_aware_fusion(
                x_se,
                self.last_liquid_probs,
                self.pair_batch_size,
                self.liquid_sampler.last_pair_band_context)
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
        x = x * gate.unsqueeze(1)
        out = x.sum(dim=2)
        if (self.liquid_output_residual_scale is not None
                and self.last_liquid_aware_delta is not None):
            residual_gate = torch.tanh(
                self.last_liquid_aware_delta).unsqueeze(1)
            out = out + self.liquid_output_residual_scale * (
                x * residual_gate).sum(dim=2)
        if return_sampling:
            return out, x, self.last_liquid_probs
        return out
