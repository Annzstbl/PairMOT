"""HSMOT R34 base config (inherits R50 architecture with R34 backbone)."""
from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r50vd_2xb4_72e_hsmot import *

pretrained = (
    'https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/'
    'master/rtdetr/resnet34vd_pretrained_f6a72dc5.pth')

model.backbone.update(
    depth=34,
    frozen_stages=-1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=False,
    init_cfg=dict(type='Pretrained', checkpoint=pretrained))
model.neck.in_channels = [128, 256, 512]
model.encoder.fpn_cfg.expansion = 0.5
model.decoder.num_layers = 4

num_blocks_list = (3, 4, 6, 3)  # r34
downsample_norm_idx_list = (2, 3, 3, 3)  # r34
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

auto_scale_lr = dict(enable=False, base_batch_size=8)
