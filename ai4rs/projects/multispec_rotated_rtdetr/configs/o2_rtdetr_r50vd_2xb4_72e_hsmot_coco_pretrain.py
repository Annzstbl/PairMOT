"""HSMOT R50: train from RT-DETR COCO backbone pretrain."""
from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r50vd_2xb4_72e_hsmot import *
    from .pretrain_paths import O2_R50_COCO_BACKBONE

load_from = None
model.backbone.init_cfg = dict(
    type='Pretrained', checkpoint=O2_R50_COCO_BACKBONE)
