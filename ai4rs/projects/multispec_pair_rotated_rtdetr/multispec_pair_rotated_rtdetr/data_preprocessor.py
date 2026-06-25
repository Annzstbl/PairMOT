# Copyright (c) AI4RS. All rights reserved.
"""Data preprocessor for pair multispec inputs ``(B, 2, C, H, W)``."""

from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.utils import is_seq_of
from mmrotate.registry import MODELS

from projects.multispec_rotated_rtdetr.multispec_rotated_rtdetr.data_preprocessor import (
    MultispecDetDataPreprocessor)


def _normalize_pair_frame(frame: torch.Tensor, mean: torch.Tensor,
                          std: torch.Tensor) -> torch.Tensor:
    """Normalize one frame ``(C, H, W)`` with arbitrary channel count."""
    frame = frame.float()
    return (frame - mean) / std


def stack_pair_batch(tensor_list: List[torch.Tensor], pad_size_divisor: int,
                     pad_value: Union[float, int]) -> torch.Tensor:
    """Pad and stack pair tensors ``(2, C, H, W)`` to ``(B, 2, C, H, W)``."""
    assert len(tensor_list) > 0
    max_h = max(t.shape[-2] for t in tensor_list)
    max_w = max(t.shape[-1] for t in tensor_list)
    pad_h = int(np.ceil(max_h / pad_size_divisor)) * pad_size_divisor
    pad_w = int(np.ceil(max_w / pad_size_divisor)) * pad_size_divisor

    padded = []
    for tensor in tensor_list:
        assert tensor.dim() == 4 and tensor.shape[0] == 2, (
            f'Each pair input must have shape (2, C, H, W), got {tensor.shape}')
        _, c, h, w = tensor.shape
        pad_bottom = pad_h - h
        pad_right = pad_w - w
        if pad_bottom > 0 or pad_right > 0:
            tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom),
                           value=pad_value)
        padded.append(tensor)
    return torch.stack(padded, dim=0)


@MODELS.register_module()
class PairMultispecDetDataPreprocessor(MultispecDetDataPreprocessor):
    """Preprocessor for HSMOT pair inputs with shape ``(2, C, H, W)`` per sample.

    Collates a batch to ``(B, 2, C, H, W)`` while reusing multispec mean/std.
    """

    def forward(self, data: dict, training: bool = False) -> dict:
        data = self.cast_data(data)
        inputs = data['inputs']
        data_samples = data.get('data_samples')

        if is_seq_of(inputs, torch.Tensor):
            batch_inputs = []
            batch_pad_shape = []
            for ori_input in inputs:
                assert ori_input.dim() == 4 and ori_input.shape[0] == 2, (
                    'PairMultispecDetDataPreprocessor expects each input to '
                    f'have shape (2, C, H, W), got {ori_input.shape}')
                frames = []
                for frame_idx in range(2):
                    frame = ori_input[frame_idx]
                    if self._enable_normalize:
                        frame = _normalize_pair_frame(frame, self.mean,
                                                      self.std)
                    else:
                        frame = frame.float()
                    frames.append(frame)
                pair = torch.stack(frames, dim=0)
                batch_inputs.append(pair)
                pad_h = int(
                    np.ceil(pair.shape[-2] / self.pad_size_divisor)
                ) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(pair.shape[-1] / self.pad_size_divisor)
                ) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
            inputs = stack_pair_batch(batch_inputs, self.pad_size_divisor,
                                      self.pad_value)
        elif isinstance(inputs, torch.Tensor):
            assert inputs.dim() == 5 and inputs.shape[1] == 2, (
                'PairMultispecDetDataPreprocessor expects batched input '
                f'(B, 2, C, H, W), got {inputs.shape}')
            inputs = inputs.float()
            if self._enable_normalize:
                mean = self.mean.view(1, 1, -1, 1, 1)
                std = self.std.view(1, 1, -1, 1, 1)
                inputs = (inputs - mean) / std
            h, w = inputs.shape[-2:]
            target_h = int(np.ceil(h / self.pad_size_divisor)
                           ) * self.pad_size_divisor
            target_w = int(np.ceil(w / self.pad_size_divisor)
                           ) * self.pad_size_divisor
            inputs = F.pad(
                inputs,
                (0, target_w - w, 0, target_h - h),
                value=self.pad_value)
            batch_pad_shape = [(target_h, target_w)] * inputs.shape[0]
        else:
            raise TypeError(
                'PairMultispecDetDataPreprocessor expects a list of tensors '
                f'or a 5D tensor, got {type(inputs)}')

        if data_samples is not None:
            batch_input_shape = tuple(inputs.shape[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape,
                })
            if self.boxtype2tensor:
                from mmdet.models.utils.misc import samplelist_boxtype2tensor
                samplelist_boxtype2tensor(data_samples)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {'inputs': inputs, 'data_samples': data_samples}
