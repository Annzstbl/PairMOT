# Copyright (c) AI4RS. All rights reserved.
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS as MDET_MODELS
from mmengine.model import BaseDataPreprocessor
from mmrotate.registry import MODELS


@MODELS.register_module()
class MultispecDetDataPreprocessor(DetDataPreprocessor):
    """DetDataPreprocessor with arbitrary input channel count.

    mmdet ``DetDataPreprocessor`` only accepts 1- or 3-channel ``mean`` /
    ``std``.  HSMOT uses 8-channel npy inputs, so mean/std tensors are
    registered directly without the RGB/gray channel assertion.
    """

    def __init__(self,
                 mean: Sequence = None,
                 std: Sequence = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None):
        BaseDataPreprocessor.__init__(self, non_blocking)
        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        assert (mean is None) == (std is None), (
            'mean and std should be both None or tuple')
        if mean is not None:
            self._enable_normalize = True
            self.register_buffer(
                'mean', torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer(
                'std', torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False
        self._channel_conversion = rgb_to_bgr or bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MDET_MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        self.pad_seg = pad_seg
        self.seg_pad_value = seg_pad_value
        self.boxtype2tensor = boxtype2tensor
