"""HSMOT R18: fine-tune from DEIM DOTA multi-scale checkpoint."""
from mmengine.config import read_base
from projects.rotated_rtdetr.rotated_rtdetr import DEIMMalLoss

with read_base():
    from .o2_rtdetr_r18vd_2xb4_72e_hsmot import *
    from .pretrain_paths import O2_DEIM_R18_DOTA_E29

load_from = O2_DEIM_R18_DOTA_E29
model.update(
    init_cfg=dict(type='Pretrained', checkpoint=O2_DEIM_R18_DOTA_E29),
    backbone=dict(
        depth=18,
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        init_cfg=dict()),
    neck=dict(in_channels=[128, 256, 512]),
    encoder=dict(fpn_cfg=dict(expansion=0.5)),
    decoder=dict(num_layers=3))
model.bbox_head.loss_cls.update(
    type=DEIMMalLoss,
    alpha=1.0,
    gamma=1.5)

num_blocks_list = (2, 2, 2, 2)  # r18
downsample_norm_idx_list = (2, 3, 3, 3)  # r18
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
custom_keys = {
    'backbone.stem.0.conv3d': dict(lr_mult=1.0),
    'backbone.stem.0.se_conv': dict(lr_mult=1.0),
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
