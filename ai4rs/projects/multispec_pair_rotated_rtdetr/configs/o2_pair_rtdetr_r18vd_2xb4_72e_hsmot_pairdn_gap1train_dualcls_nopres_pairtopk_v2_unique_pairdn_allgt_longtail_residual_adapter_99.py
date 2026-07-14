"""0713_04 long-tail residual classification adapter on local 99.

Builds on 0713_01 longtail reweight.  The classifier keeps the original logits
and adds a zero-initialized residual MLP branch, weighted toward long-tail and
fine-grained classes.  This tests whether extra class-specific nonlinear
capacity can improve cls HOTA beyond static reweighting while starting from
the same behavior as the current best long-tail baseline.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_longtail_reweight_252 import *  # noqa: F401,F403

_pairmot_root = '/data/users/wangying01/lth/PairMOT'
_gmc_root = f'{_pairmot_root}/workdir/aux/gmc_cache'

# Class order: car, bike, pedestrian, van, truck, bus, tricycle, awning-bike.
model.bbox_head.update(
    cls_residual_adapter=True,
    cls_residual_hidden_ratio=0.50,
    cls_residual_scale=0.15,
    cls_residual_weights=[
        0.15,  # car: preserve the strong head class
        0.75,  # bike
        0.25,  # pedestrian
        0.65,  # van
        1.00,  # truck
        0.90,  # bus
        1.00,  # tricycle
        0.80,  # awning-bike
    ],
)

train_dataloader.dataset.update(
    data_root=f'{_pairmot_root}/data/hsmot/train',
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1',
)
val_dataloader.dataset.update(
    data_root=f'{_pairmot_root}/data/hsmot/test',
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1',
)
test_dataloader.dataset.update(
    data_root=f'{_pairmot_root}/data/hsmot/test',
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1',
)

load_from = (
    f'{_pairmot_root}/pretrained_weights/'
    'o2_r18_hsmot_3dse_r2_e72_pair_dualcls_pairdn_adapted/'
    'pair_dualcls_pairdn_adapted_pretrain.pth')

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0713_04_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'longtail_residual_adapter_2gpu_fresh')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
