"""Formal HSMOT half-split pair RT-DETR training with PairDN."""
from mmengine.config import read_base
from mmengine.hooks.ema_hook import EMAHook
from mmdet.models.layers.ema import ExpMomentumEMA

with read_base():
    from projects.multispec_rotated_rtdetr.configs.o2_rtdetr_r18vd_2xb4_72e_hsmot import *
    from .hsmot_pair import *

from projects.multispec_rotated_rtdetr.configs.pretrain_paths import (
    O2_R18_HSMOT_3DSE_R2_E72,
)

custom_imports = dict(
    imports=[
        'projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr',
        'projects.rotated_rtdetr.rotated_rtdetr',
    ],
    allow_failed_imports=False,
)

# Created by tools/prepare_hsmot_pair_pretrain.py before the DDP launch.
# This mapping expands the single-frame cross-attention weights to both pair
# branches, which direct loading cannot do.
load_from = (
    '/data/users/litianhao01/PairMmot/pretrained_weights/'
    'o2_r18_hsmot_3dse_r2_e72_pair_adapted/pair_adapted_pretrain.pth')
find_unused_parameters = True

model.update(
    type='MultispecPairRotatedRTDETR',
    pair_mode=True,
    query_init='dual_topk',
    num_queries=300,
    dn_cfg=None,
    pair_dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=0.4,
        group_cfg=dict(dynamic=True, num_dn_queries=100),
    ),
    data_preprocessor=dict(
        type='PairMultispecDetDataPreprocessor',
        mean=hsmot_mean,
        std=hsmot_std,
        bgr_to_rgb=False,
        pad_size_divisor=32,
        boxtype2tensor=False,
        batch_augments=None,
    ),
)
model.backbone.init_cfg = None
model.backbone.se_reduction = 2
model.backbone.frozen_stages = -1
model.backbone.norm_eval = False

_pair_assigner = dict(
    type='PairHungarianAssigner',
    match_costs=[
        dict(type='mmdet.FocalLossCost', weight=2.0),
        dict(type='PairChamferCost', side='prev', weight=5.0),
        dict(type='PairChamferCost', side='curr', weight=5.0),
        dict(type='PairGDCost', side='prev', loss_type='kld', fun='log1p',
             tau=1, sqrt=False, weight=2.0),
        dict(type='PairGDCost', side='curr', loss_type='kld', fun='log1p',
             tau=1, sqrt=False, weight=2.0),
        dict(type='PairPresenceBCECost', side='prev', weight=1.0),
        dict(type='PairPresenceBCECost', side='curr', weight=1.0),
    ],
)
model.train_cfg = dict(assigner=_pair_assigner)
model.bbox_head.update(
    type='PairRotatedRTDETRHead',
    dn_loss_weight=0.2,
    loss_presence=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True,
                       loss_weight=1.0),
)
model.test_cfg = dict(max_per_img=300, rescale=False)

max_epochs = 72
train_cfg.update(max_epochs=max_epochs, val_interval=6)
default_hooks.checkpoint.update(interval=6, max_keep_ckpts=12)
default_hooks.visualization = dict(
    type='HSMOTPairValVisualizationHook',
    draw=True,
    score_thr=0.35,
    iou_thr=0.5,
    pres_thr=0.5,
    out_dir='val_vis',
    max_samples=24,
    max_samples_per_sequence=1,
    views=('deploy', 'low_score', 'iou_diag'),
)
custom_hooks = [
    dict(type='mmdet.NumClassCheckHook'),
    dict(type='PairDatasetEpochHook'),
    dict(type='TrainingCurveHook'),
    dict(
        type='EarlyStoppingHook',
        monitor='pair/independent_AP50',
        rule='greater',
        min_delta=0.002,
        patience=4,
        strict=False),
    dict(type=EMAHook, ema_type=ExpMomentumEMA, momentum=0.0001,
         update_buffers=True, priority=49),
]
work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn')
