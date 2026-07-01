"""Diagnostic same-frame overfit with GT-noised pair references.

This is a debugging upper bound only: GT references are used to initialize
queries, so this config is not a fair final comparison setting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_overfit_sameframe import *

model.query_init = 'gt_noised'
find_unused_parameters = True
