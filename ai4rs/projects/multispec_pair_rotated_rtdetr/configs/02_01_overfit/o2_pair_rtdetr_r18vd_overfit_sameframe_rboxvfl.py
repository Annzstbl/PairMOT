"""Same-frame pair overfit with rotated-IoU Varifocal quality target."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_overfit_sameframe import *

model.bbox_head.loss_cls.update(varifocal_loss_iou_type='rbox_iou')
