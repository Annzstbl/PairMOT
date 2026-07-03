# Copyright (c) AI4RS. All rights reserved.
import logging
from collections import OrderedDict
from typing import Literal

import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.logging import print_log

from mmrotate.registry import MODELS
from projects.rotated_rtdetr.rotated_rtdetr.resnet import ResNetV1dPaddle

from .pretrain_utils import (adapt_state_dict_in_channels,
                             adapt_state_dict_stem_conv3d_se,
                             load_checkpoint_state_dict)
from .stem_conv3d_se import MultispecStemConv3dSE

ExpandMode = Literal['rgbrepeat', 'interpolate']


def _filter_backbone_state_dict(state_dict: dict) -> dict:
    """Keep only backbone-compatible keys from a full detector checkpoint."""
    prefixes = ('backbone.', 'encoder.', 'decoder.', 'bbox_head.')
    if any(key.startswith(prefixes) for key in state_dict):
        backbone_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                backbone_dict[key[len('backbone.'):]] = value
        if backbone_dict:
            return backbone_dict
    return state_dict


def _filter_shape_matched_state_dict(module: nn.Module,
                                     state_dict: dict) -> OrderedDict:
    """Drop pretrained tensors whose shapes do not match ``module``."""
    current_state = module.state_dict()
    filtered = OrderedDict()
    for key, value in state_dict.items():
        if (key in current_state
                and tuple(value.shape) == tuple(current_state[key].shape)):
            filtered[key] = value
    return filtered


@MODELS.register_module()
class MultispecResNetV1dPaddle(ResNetV1dPaddle):
    """ResNetV1dPaddle backbone with multi-spectral input support.

    When loading a 3-channel pretrained checkpoint, the first conv layer is
    automatically expanded to ``in_channels`` (default 8).

    Args:
        in_channels (int): Number of input channels. Defaults to 8.
        expand_mode (str): Weight expansion strategy for pretrained loading.
            Options are ``rgbrepeat`` and ``interpolate``. Defaults to
            ``rgbrepeat``.
    """

    def __init__(self,
                 in_channels: int = 8,
                 expand_mode: ExpandMode = 'rgbrepeat',
                 **kwargs) -> None:
        self.in_channels = in_channels
        self.expand_mode = expand_mode
        super().__init__(in_channels=in_channels, **kwargs)

    def init_weights(self) -> None:
        if not isinstance(self.init_cfg, dict):
            super().init_weights()
            return

        if self.init_cfg.get('type') != 'Pretrained':
            super().init_weights()
            return

        checkpoint = self.init_cfg.get('checkpoint')
        if checkpoint is None:
            super().init_weights()
            return

        try:
            state_dict = load_checkpoint_state_dict(checkpoint)
            state_dict = _filter_backbone_state_dict(state_dict)
            state_dict = adapt_state_dict_in_channels(
                state_dict,
                in_channels=self.in_channels,
                expand_mode=self.expand_mode)
            state_dict = _filter_shape_matched_state_dict(self, state_dict)
            missing, unexpected = self.load_state_dict(
                state_dict, strict=False)
            print_log(
                f'Loaded multispec backbone from {checkpoint} with '
                f'in_channels={self.in_channels}, '
                f'expand_mode={self.expand_mode}. '
                f'missing={len(missing)}, unexpected={len(unexpected)}',
                logger='current')
        except Exception as exc:
            print_log(
                f'Failed to load multispec pretrained weights from '
                f'{checkpoint}: {exc}. Fallback to default init.',
                logger='current',
                level=logging.WARNING)
            super().init_weights()


@MODELS.register_module()
class MultispecResNetV1dPaddle3DSE(ResNetV1dPaddle):
    """ResNetV1dPaddle with 3D+SE replacement for deep-stem first 3x3 conv.

    The first ``stem.0`` Conv2d is replaced by :class:`MultispecStemConv3dSE`.
    Pretrained RGB weights from the original 3x3 conv are mapped to
    ``conv3d``; SE layers are randomly initialized.

        Args:
            in_channels (int): Number of input spectral bands. Defaults to 8.
            num_spectral (int): Spectral bands for the 3D stem. Defaults to 8.
            se_reduction (int): SE bottleneck ratio. Defaults to 4.
            liquid_sampler (dict | None): Optional Liquid Spectral Sampling
                config for the 3D stem. Defaults to None.
    """

    def __init__(self,
                 in_channels: int = 8,
                 num_spectral: int = 8,
                 se_reduction: int = 4,
                 liquid_sampler: dict = None,
                 **kwargs) -> None:
        self.num_spectral = num_spectral
        self.se_reduction = se_reduction
        self.liquid_sampler = liquid_sampler
        super().__init__(in_channels=in_channels, **kwargs)

    def _make_stem_layer(self, in_channels, stem_channels) -> None:
        if not self.deep_stem:
            super()._make_stem_layer(in_channels, stem_channels)
            return

        out_mid = stem_channels // 2
        # stem.0: same spatial config as ResNetV1d deep-stem first 3x3 conv.
        self.stem = nn.Sequential(
            MultispecStemConv3dSE(
                out_channels=out_mid,
                num_spectral=self.num_spectral,
                reduction=self.se_reduction,
                liquid_sampler=self.liquid_sampler,
            ),  # Conv3d kernel (3, 3, 3), stride (1, 2, 2), padding (1, 1, 1)
            build_norm_layer(self.norm_cfg, out_mid)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(
                self.conv_cfg,
                out_mid,
                out_mid,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, out_mid)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(
                self.conv_cfg,
                out_mid,
                stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, stem_channels)[1],
            nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def init_weights(self) -> None:
        if not isinstance(self.init_cfg, dict):
            super(ResNetV1dPaddle, self).init_weights()
            return

        if self.init_cfg.get('type') != 'Pretrained':
            super(ResNetV1dPaddle, self).init_weights()
            return

        checkpoint = self.init_cfg.get('checkpoint')
        if checkpoint is None:
            super(ResNetV1dPaddle, self).init_weights()
            return

        try:
            state_dict = load_checkpoint_state_dict(checkpoint)
            state_dict = _filter_backbone_state_dict(state_dict)
            state_dict = adapt_state_dict_stem_conv3d_se(state_dict)
            state_dict = _filter_shape_matched_state_dict(self, state_dict)
            missing, unexpected = self.load_state_dict(
                state_dict, strict=False)
            print_log(
                f'Loaded multispec 3D-SE stem backbone from {checkpoint} '
                f'with num_spectral={self.num_spectral}. '
                f'missing={len(missing)}, unexpected={len(unexpected)}',
                logger='current')
        except Exception as exc:
            print_log(
                f'Failed to load multispec 3D-SE pretrained weights from '
                f'{checkpoint}: {exc}. Fallback to default init.',
                logger='current',
                level=logging.WARNING)
            super(ResNetV1dPaddle, self).init_weights()
