# Copyright (c) AI4RS. All rights reserved.
import math
from typing import Optional, Tuple, Union

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

    The sampler outputs ``num_groups = num_spectral - spectral_kernel + 1``
    groups. Each group contains ``spectral_kernel`` softly selected spectral
    bands. The head bias is initialized to fixed adjacent windows, so the
    initial sampling pattern is equivalent to unpadded Conv3d windows.
    """

    def __init__(self,
                 num_spectral: int = 8,
                 spectral_kernel: int = STEM_SPECTRAL_KERNEL,
                 embed_dims: int = 32,
                 tau: float = 1.0,
                 hard: bool = False,
                 init_logit: float = 8.0,
                 head_weight_std: float = 0.0,
                 deterministic_eval: bool = True,
                 eval_hard: bool = True,
                 lowres_grad_size: Optional[Union[int, Tuple[int, int]]] = None,
                 lowres_grad_downsample: int = 4,
                 use_lowres_grad_correction: bool = True) -> None:
        super().__init__()
        assert num_spectral >= spectral_kernel
        self.num_spectral = num_spectral
        self.spectral_kernel = spectral_kernel
        self.num_groups = num_spectral - spectral_kernel + 1
        self.embed_dims = embed_dims
        self.tau = tau
        self.hard = hard
        self.deterministic_eval = deterministic_eval
        self.eval_hard = eval_hard
        self.lowres_grad_size = lowres_grad_size
        self.lowres_grad_downsample = lowres_grad_downsample
        self.use_lowres_grad_correction = use_lowres_grad_correction

        self.desc_proj = nn.Linear(3, embed_dims)
        self.band_embedding = nn.Parameter(torch.zeros(num_spectral, embed_dims))
        self.w1 = nn.Linear(embed_dims * 2, embed_dims)
        self.w2 = nn.Linear(embed_dims * 2, embed_dims)
        self.head = nn.Linear(
            embed_dims, self.num_groups * spectral_kernel * num_spectral)
        self._init_weights(init_logit, head_weight_std)

    def _init_weights(self, init_logit: float, head_weight_std: float) -> None:
        nn.init.zeros_(self.band_embedding)
        nn.init.xavier_uniform_(self.desc_proj.weight)
        nn.init.zeros_(self.desc_proj.bias)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)
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
                bias[group_idx, kernel_idx, group_idx + kernel_idx] = init_logit
        with torch.no_grad():
            self.head.bias.copy_(bias.reshape(-1))

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training or not self.deterministic_eval:
            return F.gumbel_softmax(
                logits, tau=self.tau, hard=self.hard, dim=-1)

        probs = F.softmax(logits / self.tau, dim=-1)
        if not self.eval_hard:
            return probs
        indices = probs.argmax(dim=-1, keepdim=True)
        hard_probs = torch.zeros_like(probs).scatter_(-1, indices, 1.0)
        return hard_probs - probs.detach() + probs

    def _lowres_size(self, height: int, width: int) -> Tuple[int, int]:
        if self.lowres_grad_size is None:
            downsample = max(1, self.lowres_grad_downsample)
            return max(1, height // downsample), max(1, width // downsample)
        if isinstance(self.lowres_grad_size, int):
            size = self.lowres_grad_size
            return min(size, height), min(size, width)
        return min(self.lowres_grad_size[0], height), min(
            self.lowres_grad_size[1], width)

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
        correction = F.interpolate(
            lowres_correction,
            size=(height, width),
            mode='bilinear',
            align_corners=False).view(
                batch_size, self.num_groups, self.spectral_kernel, height,
                width)
        return sampled + correction

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim == 4, f'Expected [B, S, H, W], got {tuple(x.shape)}'
        assert x.size(1) == self.num_spectral, (
            f'Expected {self.num_spectral} spectral bands, got {x.size(1)}')

        mean = x.mean(dim=(-2, -1))
        std = x.flatten(2).std(dim=-1)
        maxv = x.amax(dim=(-2, -1))
        desc = torch.stack([mean, std, maxv], dim=-1)
        desc = self.desc_proj(desc) + self.band_embedding.unsqueeze(0)

        hidden = desc.new_zeros(desc.size(0), self.embed_dims)
        for band_idx in range(self.num_spectral):
            cell_input = torch.cat([desc[:, band_idx], hidden], dim=-1)
            h_hat = torch.tanh(self.w1(cell_input))
            alpha = torch.sigmoid(self.w2(cell_input))
            hidden = alpha * hidden + (1 - alpha) * h_hat

        logits = self.head(hidden).view(
            x.size(0), self.num_groups, self.spectral_kernel,
            self.num_spectral)
        probs = self._sample(logits)
        sampled = self._sample_bands(x, probs)
        return sampled, probs


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
        self.conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=(spectral_kernel, spatial_kernel, spatial_kernel),
            stride=(1, spatial_stride, spatial_stride),
            padding=(spectral_padding, spatial_padding, spatial_padding),
            bias=False)

        if self.use_liquid_sampler:
            sampler_cfg = dict(liquid_sampler)
            sampler_cfg.setdefault('num_spectral', num_spectral)
            sampler_cfg.setdefault('spectral_kernel', spectral_kernel)
            self.liquid_sampler = LiquidSpectralSampler(**sampler_cfg)
            temporal_output_size = self.liquid_sampler.num_groups
        else:
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
        self.last_liquid_groups = None
        self.last_liquid_probs = None
        self._init_se_weights()

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
        sampled, probs = self.liquid_sampler(x)
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
        return groups

    def forward(self,
                x: torch.Tensor,
                return_sampling: bool = False):
        if self.use_liquid_sampler:
            x = self._forward_liquid(x)
        else:
            x = self._forward_fixed(x)

        x_se = x.mean(dim=1)
        gate = torch.sigmoid(self.se_conv2(F.relu(self.se_conv1(x_se))))
        x = x * gate.unsqueeze(1)
        out = x.sum(dim=2)
        if return_sampling:
            return out, x, self.last_liquid_probs
        return out
