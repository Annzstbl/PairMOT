"""Same-frame pair overfit with relaxed gradient clipping.

Diagnostic/minimal optimization change after the fair baseline failed to
overfit.  Pretraining, data, model, query init, and batch shape are unchanged.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_overfit_sameframe import *

optim_wrapper.clip_grad = dict(max_norm=1.0, norm_type=2)
