# Copyright (c) AI4RS. All rights reserved.
"""Tensor helpers for the PairMOT long-edge box representation."""

from __future__ import annotations

import torch
from torch import Tensor


def canonicalize_le180_start0(boxes: Tensor) -> Tensor:
    """Canonicalize ``(cx, cy, w, h, theta)`` to long-edge ``[0, pi)``.

    The operation is differentiable away from the unavoidable ``w == h`` and
    angle-wrap boundaries. It intentionally preserves PairMOT's established
    internal representation instead of converting boxes to MMRotate le135.
    """
    if boxes.size(-1) != 5:
        raise ValueError(f'Expected (..., 5) boxes, got {tuple(boxes.shape)}')
    center = boxes[..., :2]
    width = boxes[..., 2]
    height = boxes[..., 3]
    angle = boxes[..., 4]
    swap = width < height
    long_edge = torch.where(swap, height, width)
    short_edge = torch.where(swap, width, height)
    angle = torch.remainder(
        angle + swap.to(angle.dtype) * (torch.pi / 2), torch.pi)
    return torch.cat([
        center,
        long_edge.unsqueeze(-1),
        short_edge.unsqueeze(-1),
        angle.unsqueeze(-1),
    ], dim=-1)


def encode_le180_l1(boxes: Tensor, factors: Tensor,
                    angle_weight: float) -> Tensor:
    """Encode normalized boxes for image-balanced parameter-space L1.

    Spatial coordinates are measured in physical pixels and divided by the
    image geometric-mean scale. The angle term is normalized by the configured
    180-degree factor and multiplied by one fixed, target-independent weight.
    """
    if boxes.shape != factors.shape or boxes.size(-1) != 5:
        raise ValueError(
            f'boxes and factors must both be (N, 5), got '
            f'{tuple(boxes.shape)} and {tuple(factors.shape)}')
    physical = canonicalize_le180_start0(boxes * factors)
    image_scale = torch.sqrt(factors[:, 0] * factors[:, 1]).unsqueeze(-1)
    spatial = physical[:, :4] / image_scale
    angle = (physical[:, 4:5] / factors[:, 4:5]) * float(angle_weight)
    return torch.cat([spatial, angle], dim=-1)
