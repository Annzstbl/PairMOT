"""HSMOT R18: fine-tune from full O2-RTDETR DOTA checkpoint."""
from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r18vd_2xb4_72e_hsmot import *
    from .pretrain_paths import O2_R18_DOTA_E72

load_from = O2_R18_DOTA_E72
model.backbone.init_cfg = dict(
    type='Pretrained', checkpoint=O2_R18_DOTA_E72)
