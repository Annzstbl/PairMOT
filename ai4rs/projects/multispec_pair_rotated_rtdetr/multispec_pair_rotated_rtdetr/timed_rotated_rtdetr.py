# Copyright (c) AI4RS. All rights reserved.
"""Timed single-frame Rotated RT-DETR for profiling overfit runs."""

from __future__ import annotations

from typing import Dict

from mmdet.structures import OptSampleList
from mmrotate.registry import MODELS
from torch import Tensor

from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETR

from .component_timer import CudaComponentTimer


@MODELS.register_module()
class TimedRotatedRTDETR(RotatedRTDETR):
    """Rotated RT-DETR with per-component training timing.

    This keeps the normal single-frame RT-DETR forward/loss path intact and
    only records the same timing fields consumed by ``PairComponentTimerHook``.
    """

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: OptSampleList) -> Dict:
        timer = CudaComponentTimer()
        try:
            img_feats = timer.record(
                'backbone_neck',
                lambda: self.extract_feat(batch_inputs))
            encoder_inputs_dict, decoder_inputs_dict = timer.record(
                'pre_transformer',
                lambda: self.pre_transformer(img_feats, batch_data_samples))
            encoder_outputs_dict = timer.record(
                'encoder',
                lambda: self.forward_encoder(**encoder_inputs_dict))
            tmp_dec_in, head_inputs_dict = timer.record(
                'pre_decoder',
                lambda: self.pre_decoder(
                    batch_data_samples=batch_data_samples,
                    **encoder_outputs_dict))
            decoder_inputs_dict.update(tmp_dec_in)
            decoder_outputs_dict = timer.record(
                'decoder',
                lambda: self.forward_decoder(**decoder_inputs_dict))
            head_inputs_dict.update(decoder_outputs_dict)
            losses = timer.record(
                'head_loss',
                lambda: self.bbox_head.loss(
                    **head_inputs_dict,
                    batch_data_samples=batch_data_samples))
        finally:
            self._last_component_timings = timer.get_durations()
        return losses
