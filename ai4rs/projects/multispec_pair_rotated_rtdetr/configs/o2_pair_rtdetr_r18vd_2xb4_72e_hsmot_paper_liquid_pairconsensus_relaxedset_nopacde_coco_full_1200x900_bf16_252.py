"""0719_03 pair-consensus Liquid without PACDE on server 252."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_pairconsensus_relaxedset_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


# Strict PACDE ablation: preserve shared routing, relaxed Set-Transport and
# pair-aligned fusion from 0719_01 while removing only compact-detail gating.
model['backbone']['liquid_sampler'][
    'pair_aligned_compact_detail_enhancement'] = None
optim_wrapper['paramwise_cfg']['custom_keys'].pop(
    'backbone.stem.0.pair_aligned_compact_detail_enhancement', None)

_pairmot_root = '/data/users/litianhao01/PairMmot'
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
    '/data4/litianhao/PairMmot/workdir_252/'
    '0719_03_paper_liquid_pairconsensus_relaxedset_nopacde_r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
