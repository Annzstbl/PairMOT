"""0723_01: 0718_01 with representation-consistent paired DN."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_independent_diffproduct_accuracyfix_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


_pairmot_root = '/data/users/wangying01/lth/PairMOT'
_hsmot_root = f'{_pairmot_root}/data/hsmot'
_gmc_root = f'{_pairmot_root}/workdir/aux/gmc_cache'

train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1')
val_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/test',
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1')
test_dataloader = val_dataloader

# Preserve the 0718_01 model and training protocol. These values make the new
# PairDN distribution explicit instead of relying on generator defaults.
model['pair_dn_cfg'].update(
    positive_hard_ratio=0.75,
    positive_hard_min_magnitude=0.5,
    positive_hard_max_magnitude=1.25,
    negative_ratio=0.5,
    negative_min_magnitude=0.75,
    negative_max_magnitude=1.5,
    negative_max_iou=0.4,
    negative_resample_attempts=4)
model['bbox_head'].update(bbox_angle_l1_weight=0.05)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0723_01_paper_liquid_independent_diffproduct_pairdn_paircoherent_le180_'
    'r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
