"""R18 2D spectral-interpolate COCO pretrain."""
from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r18vd_2xb4_72e_hsmot_coco_pretrain_rgbrepeat2d import *

model.backbone.expand_mode = 'interpolate'
