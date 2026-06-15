# Copyright (c) AI4RS. All rights reserved.
import math
from typing import Dict, Literal, Optional, Tuple

import torch


ExpandMode = Literal['rgbrepeat', 'interpolate']

HSMOT_SPECTRAL_BANDS = [422.5, 487.5, 550, 602.5, 660, 725, 785, 887.5]
RGB_BANDS = dict(R=700.0, G=546.1, B=435.8)


def expand_conv1_weight(
        weight: torch.Tensor,
        in_channels: int = 8,
        expand_mode: ExpandMode = 'rgbrepeat') -> torch.Tensor:
    """Expand the first conv weight from 3 input channels to ``in_channels``.

    Args:
        weight (Tensor): Shape ``(out_channels, 3, k, k)``.
        in_channels (int): Target input channels. Defaults to 8.
        expand_mode (str): ``rgbrepeat`` repeats RGB weights cyclically;
            ``interpolate`` linearly interpolates along spectral bands.
    """
    if weight.ndim != 4:
        raise ValueError(f'Expected 4D conv weight, got shape {weight.shape}')
    if weight.shape[1] == in_channels:
        return weight
    if weight.shape[1] != 3:
        raise ValueError(
            f'Only 3-channel expansion is supported, got in_channels='
            f'{weight.shape[1]}')

    if expand_mode == 'rgbrepeat':
        repeat_times = math.ceil(in_channels / 3)
        expanded = weight.repeat(1, repeat_times, 1, 1)
        return expanded[:, :in_channels, :, :].contiguous()

    if expand_mode == 'interpolate':
        if in_channels != 8:
            raise ValueError(
                'interpolate mode currently supports 8 channels only.')
        r_band = RGB_BANDS['R']
        g_band = RGB_BANDS['G']
        b_band = RGB_BANDS['B']
        r_weight = weight[:, 0, :, :]
        g_weight = weight[:, 1, :, :]
        b_weight = weight[:, 2, :, :]
        channel_weights = []
        for band in HSMOT_SPECTRAL_BANDS:
            if band <= g_band:
                w = (b_weight * (g_band - band) +
                     g_weight * (band - b_band)) / (g_band - b_band)
            else:
                w = (g_weight * (r_band - band) +
                     r_weight * (band - g_band)) / (r_band - g_band)
            channel_weights.append(w.unsqueeze(1))
        return torch.cat(channel_weights, dim=1).contiguous()

    raise ValueError(f'Unsupported expand_mode: {expand_mode}')


def _find_first_conv_keys(state_dict: Dict[str, torch.Tensor]) -> Tuple[str, ...]:
    """Return the first input conv key only (deep-stem or classic ResNet)."""
    for key in state_dict:
        if key == 'stem.0.weight' or key.endswith('.stem.0.weight'):
            return (key,)
    for key in state_dict:
        if key.endswith('conv1.weight') and '.layer' not in key:
            return (key,)
    raise KeyError(
        'Could not find first conv weight (stem.0.weight or conv1.weight) '
        'in checkpoint.')


def _find_deep_stem_conv_key(state_dict: Dict[str, torch.Tensor]) -> str:
    """Return deep-stem first conv key for Conv2d -> Conv3d remap."""
    for key in state_dict:
        if key == 'stem.0.weight' or key.endswith('.stem.0.weight'):
            return key
    raise KeyError(
        'Could not find stem.0.weight for Conv3d stem remap in checkpoint.')


def convert_stem_conv2d_to_conv3d_weight(
        weight: torch.Tensor,
        spatial_kernel: int = 3,
        spectral_kernel: int = 3) -> torch.Tensor:
    """Map deep-stem first Conv2d weight to Conv3d stem weight.

    ResNetV1d ``stem.0`` is ``Conv2d(3, out, kernel_size=3, stride=2)``.
    The corresponding 3D stem uses ``Conv3d(1, out, kernel_size=(3, 3, 3),
    stride=(1, 2, 2))``. RGB input channels ``(out, 3, 3, 3)`` are mapped
    to the spectral kernel axis, producing ``(out, 1, 3, 3, 3)``.
    """
    if weight.ndim == 5:
        expected = (weight.shape[0], 1, spectral_kernel, spatial_kernel,
                    spatial_kernel)
        if tuple(weight.shape) != expected:
            raise ValueError(
                f'5D weight shape {tuple(weight.shape)} != expected {expected}')
        return weight
    if weight.ndim != 4:
        raise ValueError(
            f'Expected 4D or 5D conv weight, got shape {weight.shape}')
    _, in_channels, kh, kw = weight.shape
    if kh != spatial_kernel or kw != spatial_kernel:
        raise ValueError(
            f'Expected spatial kernel {spatial_kernel}x{spatial_kernel}, '
            f'got {kh}x{kw}')
    if in_channels == spectral_kernel:
        return weight.unsqueeze(1).contiguous()
    if in_channels > spectral_kernel:
        return weight[:, :spectral_kernel, :, :].unsqueeze(1).contiguous()
    raise ValueError(
        f'Expected at least {spectral_kernel} input channels, '
        f'got in_channels={in_channels}')


def adapt_state_dict_in_channels(
        state_dict: Dict[str, torch.Tensor],
        in_channels: int = 8,
        expand_mode: ExpandMode = 'rgbrepeat') -> Dict[str, torch.Tensor]:
    """Adapt checkpoint state dict for multi-spectral input."""
    adapted = state_dict.copy()
    for key in _find_first_conv_keys(adapted):
        adapted[key] = expand_conv1_weight(
            adapted[key], in_channels=in_channels, expand_mode=expand_mode)
    return adapted


def adapt_state_dict_stem_conv3d_se(
        state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap deep-stem first Conv2d weight to ``stem.0.conv3d.weight``."""
    adapted = state_dict.copy()
    key = _find_deep_stem_conv_key(adapted)
    weight = adapted.pop(key)
    stem_prefix = key.rsplit('.', 1)[0]
    adapted[f'{stem_prefix}.conv3d.weight'] = (
        convert_stem_conv2d_to_conv3d_weight(weight))
    return adapted


def load_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load a checkpoint and return its state dict."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            return checkpoint['model']
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f'Unsupported checkpoint format: {checkpoint_path}')
