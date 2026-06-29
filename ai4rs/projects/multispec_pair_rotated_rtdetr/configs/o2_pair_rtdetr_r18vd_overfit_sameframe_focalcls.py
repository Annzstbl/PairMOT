"""Same-frame pair overfit with positive-target sigmoid FocalLoss cls."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_overfit_sameframe import *

model.bbox_head.loss_cls = dict(
    type='mmdet.FocalLoss',
    use_sigmoid=True,
    alpha=0.25,
    gamma=2.0,
    loss_weight=1.0)
