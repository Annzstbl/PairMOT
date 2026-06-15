"""R18 2D rgb-repeat COCO pretrain with a higher backbone LR."""
from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r18vd_2xb4_72e_hsmot_coco_pretrain_rgbrepeat2d import *

for key, value in custom_keys.items():
    if key == 'backbone':
        value['lr_mult'] = 0.2
    elif key.startswith('backbone.layer'):
        value['lr_mult'] = 0.2
custom_keys['backbone.stem.0'] = dict(lr_mult=1.0)
optim_wrapper.paramwise_cfg = dict(
    custom_keys=custom_keys,
    norm_decay_mult=0,
    bypass_duplicate=True)
