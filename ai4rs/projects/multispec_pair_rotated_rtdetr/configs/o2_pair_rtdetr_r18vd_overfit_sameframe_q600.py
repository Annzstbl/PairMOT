"""Same-frame pair overfit with more top-k pair queries.

Minimal capacity diagnostic: keep the same model family, optimizer, data and
pretrain adaptation, but increase pair decoder queries from 300 to 600.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_overfit_sameframe import *

model.num_queries = 600
model.test_cfg = dict(max_per_img=600, rescale=False)
