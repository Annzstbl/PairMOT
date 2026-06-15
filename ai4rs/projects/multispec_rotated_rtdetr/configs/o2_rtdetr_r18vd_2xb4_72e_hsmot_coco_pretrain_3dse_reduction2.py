"""R18 3D-SE COCO pretrain with a wider SE bottleneck."""
from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r18vd_2xb4_72e_hsmot_coco_pretrain import *

model.backbone.se_reduction = 2
custom_keys['backbone.stem.0.conv3d'] = dict(lr_mult=1.0)
custom_keys['backbone.stem.0.se_conv'] = dict(lr_mult=1.0)
optim_wrapper.paramwise_cfg = dict(
    custom_keys=custom_keys,
    norm_decay_mult=0,
    bypass_duplicate=True)
