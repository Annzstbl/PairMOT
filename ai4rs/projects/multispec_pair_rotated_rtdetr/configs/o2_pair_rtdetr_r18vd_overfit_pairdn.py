"""Pair RT-DETR overfit variant with track-union PairDN enabled."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_overfit import *

# PairDN is separate from the single-frame CdnQueryGenerator.  It creates one
# shared class query and two noisy rotated references for each pair GT row.
model.pair_dn_cfg = dict(
    label_noise_scale=0.5,
    box_noise_scale=0.4,
    group_cfg=dict(dynamic=True, num_dn_queries=100))
model.bbox_head.dn_loss_weight = 0.2
