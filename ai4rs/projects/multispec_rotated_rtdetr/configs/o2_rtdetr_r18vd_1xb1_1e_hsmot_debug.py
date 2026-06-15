"""1-epoch debug config on synthetic minimal HSMOT (npy)."""
from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r18vd_2xb4_72e_hsmot import *
    from .hsmot_debug import *

model.num_queries = 50
model.backbone.init_cfg = None
model.backbone.frozen_stages = 4
model.backbone.norm_eval = True
model.dn_cfg.group_cfg.num_dn_queries = 20

max_epochs = 1
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=10)
]
custom_hooks = []
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook', draw=False))

work_dir = 'work_dirs/o2_rtdetr_r18vd_1xb1_1e_hsmot_debug'
