"""Same-frame pair overfit with geometry-only Hungarian assignment.

Diagnostic config: keep the model, optimizer, data and pretrain path identical
to the same-frame overfit config, but remove classification cost from matching.
This checks whether score/geometry mismatch is caused by unstable cls-driven
assignment during small-set overfitting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_overfit_sameframe import *

model.train_cfg['assigner']['match_costs'][0]['weight'] = 0.0
