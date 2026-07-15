"""Minimal spectral Conv3d + pixel-wise SE stem for CTracker."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform_gate_logit(num_bands):
    if num_bands <= 1:
        raise ValueError('num_bands must be greater than one')
    return math.log(1.0 / (num_bands - 1))


class SpectralStemConv3dSE(nn.Module):
    """Preserve CTracker's 7x7 spatial stem and fuse spectral responses.

    An input ``[B, S, H, W]`` is interpreted as ``[B, 1, S, H, W]``.
    The spectral kernel is three so an RGB ResNet conv1 weight can be mapped
    exactly from ``[C, 3, 7, 7]`` to ``[C, 1, 3, 7, 7]``.
    """

    def __init__(self, out_channels=64, num_spectral=8, reduction=4):
        super().__init__()
        if num_spectral < 3:
            raise ValueError('num_spectral must be at least three')
        hidden = num_spectral // reduction
        if hidden < 1:
            raise ValueError('SE reduction produces an empty bottleneck')
        self.num_spectral = num_spectral
        self.conv3d = nn.Conv3d(
            1,
            out_channels,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        self.se_conv1 = nn.Conv2d(num_spectral, hidden, 3, padding=1)
        self.se_conv2 = nn.Conv2d(hidden, num_spectral, 3, padding=1)
        self.reset_se_parameters()

    def reset_se_parameters(self):
        nn.init.zeros_(self.se_conv1.weight)
        nn.init.zeros_(self.se_conv1.bias)
        nn.init.zeros_(self.se_conv2.weight)
        nn.init.constant_(self.se_conv2.bias,
                          uniform_gate_logit(self.num_spectral))

    def load_rgb_weight(self, weight):
        expected = (self.conv3d.out_channels, 3, 7, 7)
        if tuple(weight.shape) != expected:
            raise ValueError(
                f'Expected RGB conv1 weight {expected}, got {tuple(weight.shape)}')
        with torch.no_grad():
            self.conv3d.weight.copy_(weight.unsqueeze(1))

    def forward(self, x, return_gate=False):
        if x.ndim != 4 or x.size(1) != self.num_spectral:
            raise ValueError(
                f'Expected [B,{self.num_spectral},H,W], got {tuple(x.shape)}')
        responses = self.conv3d(x.unsqueeze(1))
        descriptor = responses.mean(dim=1)
        gate = torch.sigmoid(
            self.se_conv2(F.relu(self.se_conv1(descriptor), inplace=True)))
        output = (responses * gate.unsqueeze(1)).sum(dim=2)
        if return_gate:
            return output, gate
        return output
