"""HSMOT R18 COCO pretrain with 2D rgb-repeat multispectral stem.

This keeps the original ResNetV1d deep-stem Conv2d path and expands the
RGB pretrained first-conv weights to 8 HSMOT channels by cyclic RGB repeat.
"""
from mmengine.config import read_base

from projects.multispec_rotated_rtdetr.multispec_rotated_rtdetr import (
    MultispecResNetV1dPaddle)

with read_base():
    from .o2_rtdetr_r18vd_2xb4_72e_hsmot_coco_pretrain import *

model.backbone.update(
    type=MultispecResNetV1dPaddle,
    expand_mode='rgbrepeat')
del model.backbone['num_spectral']
del model.backbone['se_reduction']

num_blocks_list = (2, 2, 2, 2)  # r18
downsample_norm_idx_list = (2, 3, 3, 3)  # r18
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
custom_keys = {
    'backbone.stem.0': dict(lr_mult=1.0),
    'backbone': dict(lr_mult=0.1),
}
custom_keys.update({
    f'backbone.layer{stage_id + 1}.{block_id}.bn': backbone_norm_multi
    for stage_id, num_blocks in enumerate(num_blocks_list)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.layer{stage_id + 1}.{block_id}.downsample.'
    f'{downsample_norm_idx - 1}': backbone_norm_multi
    for stage_id, (num_blocks, downsample_norm_idx) in enumerate(
        zip(num_blocks_list, downsample_norm_idx_list))
    for block_id in range(num_blocks)
})

optim_wrapper.paramwise_cfg = dict(
    custom_keys=custom_keys,
    norm_decay_mult=0,
    bypass_duplicate=True)
