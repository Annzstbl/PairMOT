"""0719_04 paper-protocol replay of the historical 0711_01 Liquid core."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


# Keep the historically effective fusion-centric structure: independent
# per-frame routing, wide pattern-aware LAF, and coverage-aware GroupMod.
_sampler = model['backbone']['liquid_sampler']
_sampler.update(
    pair_sampler_router=None,
    hard_group_unique_sets=False,
    soft_group_set_transport=None,
)
_sampler['liquid_aware_fusion'].update(pair_transport=None)

_pairmot_root = '/data/users/litianhao/PairMOT'
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
    '/data4/litianhao/PairMmot/workdir_197/'
    '0719_04_paper_liquid_widelaf_groupmod_r18_coco_full_1200x900_bf16_'
    'orderedpairs_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
