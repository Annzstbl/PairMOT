"""Same-frame pair overfit with iter-based LR decay.

Optimization-only diagnostic: keep the strongest q300/fusion-average model path
and reduce LR late in the overfit run to refine duplicate/near-miss queries.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_overfit_sameframe import *

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=100),
    dict(
        type='MultiStepLR',
        by_epoch=False,
        begin=100,
        end=6000,
        milestones=[3000, 4500],
        gamma=0.1),
]
