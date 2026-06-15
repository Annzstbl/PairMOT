# Copyright (c) AI4RS. All rights reserved.
import math

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
    """

    def __init__(self,
                 out_channels: int,
                 num_spectral: int = 8,
                 spectral_kernel: int = STEM_SPECTRAL_KERNEL,
                 spatial_kernel: int = STEM_SPATIAL_KERNEL,
                 spatial_stride: int = STEM_SPATIAL_STRIDE,
                 reduction: int = 4) -> None:
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
        self.conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=(spectral_kernel, spatial_kernel, spatial_kernel),
            stride=(1, spatial_stride, spatial_stride),
            padding=(spectral_padding, spatial_padding, spatial_padding),
            bias=False)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)

        x = self.conv3d(x)
        x_se = x.mean(dim=1)
        gate = torch.sigmoid(self.se_conv2(F.relu(self.se_conv1(x_se))))
        x = x * gate.unsqueeze(1)
        return x.sum(dim=2)
