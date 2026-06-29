"""Continue interrupted formal half run from epoch 66 weights for 6 epochs."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn import *

load_from = (
    '/data/users/litianhao01/PairMmot/workdir/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn/epoch_66.pth')

max_epochs = 6
train_cfg.update(max_epochs=max_epochs, val_interval=6)
default_hooks.checkpoint.update(interval=6, max_keep_ckpts=3)
param_scheduler = []

custom_hooks = [
    dict(type='mmdet.NumClassCheckHook'),
    dict(type='PairDatasetEpochHook'),
    dict(type='TrainingCurveHook'),
]

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_cont_epoch66_to72')
