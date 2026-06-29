"""Resume helper for interrupted formal half run without EMAHook."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn import *

custom_hooks = [
    dict(type='mmdet.NumClassCheckHook'),
    dict(type='PairDatasetEpochHook'),
    dict(type='TrainingCurveHook'),
]
