"""Pair RT-DETR same-frame overfit: prev/curr identical + InfiniteSampler."""
from mmengine.config import read_base
from mmengine.runner.loops import IterBasedTrainLoop, ValLoop
from mmengine.optim.optimizer import OptimWrapper
from torch.optim.adamw import AdamW

from projects.multispec_rotated_rtdetr.configs.pretrain_paths import (
    O2_R18_HSMOT_3DSE_R2_E72,
)

with read_base():
    from projects.multispec_rotated_rtdetr.configs.o2_rtdetr_r18vd_2xb4_72e_hsmot import *
    from .hsmot_pair_overfit_sameframe import *

custom_imports = dict(
    imports=[
        'projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr',
        'projects.rotated_rtdetr.rotated_rtdetr',
    ],
    allow_failed_imports=False)

load_from = O2_R18_HSMOT_3DSE_R2_E72
pair_pretrain_adapt = True

# dual_topk leaves learned query/ref embeddings unused under DDP.
find_unused_parameters = True

model.update(
    type='MultispecPairRotatedRTDETR',
    pair_mode=True,
    query_init='dual_topk',
    num_queries=300,
    dn_cfg=None,
    data_preprocessor=dict(
        type='PairMultispecDetDataPreprocessor',
        mean=hsmot_mean,
        std=hsmot_std,
        bgr_to_rgb=False,
        pad_size_divisor=32,
        boxtype2tensor=False,
        batch_augments=None),
)
model.backbone.init_cfg = None
model.backbone.se_reduction = 2
model.backbone.frozen_stages = -1
model.backbone.norm_eval = False

custom_keys['backbone.stem.0.conv3d'] = dict(lr_mult=1.0)
custom_keys['backbone.stem.0.se_conv'] = dict(lr_mult=1.0)

_pair_assigner = dict(
    type='PairHungarianAssigner',
    match_costs=[
        dict(type='mmdet.FocalLossCost', weight=2.0),
        dict(type='PairChamferCost', side='prev', weight=5.0),
        dict(type='PairChamferCost', side='curr', weight=5.0),
        dict(
            type='PairGDCost',
            side='prev',
            loss_type='kld',
            fun='log1p',
            tau=1,
            sqrt=False,
            weight=2.0),
        dict(
            type='PairGDCost',
            side='curr',
            loss_type='kld',
            fun='log1p',
            tau=1,
            sqrt=False,
            weight=2.0),
        dict(type='PairPresenceBCECost', side='prev', weight=1.0),
        dict(type='PairPresenceBCECost', side='curr', weight=1.0),
    ])
model.train_cfg = dict(assigner=_pair_assigner)

model.bbox_head.update(
    type='PairRotatedRTDETRHead',
    loss_presence=dict(
        type='mmdet.CrossEntropyLoss',
        use_sigmoid=True,
        loss_weight=1.0),
)
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
val_interval = 200
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
        type='HSMOTPairValVisualizationHook',
        draw=True,
        score_thr=0.35,
        iou_thr=0.5,
        pres_thr=0.5,
        out_dir='val_vis',
        diagnostic_mode=True))

custom_hooks = [
    dict(type='PairComponentTimerHook', interval=50),
]
work_dir = 'work_dirs/o2_pair_rtdetr_r18vd_overfit_sameframe'
