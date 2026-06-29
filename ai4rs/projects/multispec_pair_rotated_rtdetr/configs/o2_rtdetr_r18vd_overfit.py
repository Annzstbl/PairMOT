"""Single-frame RT-DETR overfit config: fixed mini HSMOT frames, weak aug."""
from mmengine.config import read_base
from mmengine.runner.loops import IterBasedTrainLoop, TestLoop, ValLoop
from mmengine.optim.optimizer import OptimWrapper
from torch.optim.adamw import AdamW

from projects.multispec_rotated_rtdetr.configs.pretrain_paths import (
    O2_R18_HSMOT_3DSE_R2_E72,
)

with read_base():
    from projects.multispec_rotated_rtdetr.configs.o2_rtdetr_r18vd_2xb4_72e_hsmot_coco_pretrain_3dse_reduction2 import *
    from .hsmot_single_overfit import *

custom_imports = dict(
    imports=[
        'projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr',
        'projects.multispec_rotated_rtdetr.multispec_rotated_rtdetr',
        'projects.rotated_rtdetr.rotated_rtdetr',
    ],
    allow_failed_imports=False)

load_from = O2_R18_HSMOT_3DSE_R2_E72

find_unused_parameters = False

model.update(type='TimedRotatedRTDETR')
# model.update(
    # num_queries=50,
# )
# model.dn_cfg.group_cfg.num_dn_queries = 20
model.backbone.init_cfg = None
model.backbone.se_reduction = 2
model.backbone.frozen_stages = -1
model.backbone.norm_eval = False

custom_keys['backbone.stem.0.conv3d'] = dict(lr_mult=1.0)
custom_keys['backbone.stem.0.se_conv'] = dict(lr_mult=1.0)

model.test_cfg = dict(max_per_img=300, rescale=False)

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=custom_keys,
        norm_decay_mult=0,
        bypass_duplicate=True))

from projects.multispec_rotated_rtdetr.configs.logging_overfit import (
    console_suppress_overfit,
)

console_suppress_patterns = console_suppress_overfit

max_iters = 3000
val_interval = 500
train_cfg = dict(
    type=IterBasedTrainLoop, max_iters=max_iters, val_interval=val_interval)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=100)
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='HSMOTSingleValVisualizationHook',
        draw=True,
        score_thr=0.35,
        iou_thr=0.5,
        out_dir='val_vis'))

custom_hooks = [
    dict(type='PairComponentTimerHook', interval=50),
]

work_dir = 'work_dirs/o2_rtdetr_r18vd_overfit'
