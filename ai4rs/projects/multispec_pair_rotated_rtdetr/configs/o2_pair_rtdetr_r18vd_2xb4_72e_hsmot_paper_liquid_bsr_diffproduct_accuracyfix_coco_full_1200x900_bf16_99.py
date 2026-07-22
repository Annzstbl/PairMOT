"""0721_03 BSR-Liquid: blockwise spectral recurrent routing."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_independent_diffproduct_accuracyfix_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


# Replace the global per-band mean/std/max descriptor. Each 75x75 block forms
# an independent eight-band recurrent sequence; block hidden mean/std/max then
# produce the single image-level route state consumed by the existing head.
model['backbone']['liquid_sampler']['block_route_descriptor'] = dict(
    grid_size=(12, 16))

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

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0721_03_paper_liquid_bsr_diffproduct_accuracyfix_r18_coco_full_'
    '1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
