"""0723_03: 0723_01 PairDN model with DSE fusion evidence."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_independent_diffproduct_pairdn_paircoherent_le180_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


model['backbone']['liquid_sampler'][
    'dispersion_aware_spectral_evidence'] = dict(eps=1e-6)
model['pair_dn_cfg']['share_pair_noise'] = True
optim_wrapper['paramwise_cfg']['custom_keys'][
    'backbone.stem.0.dispersion_aware_spectral_evidence'] = dict(
        lr_mult=1.0)

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
    '0723_03_paper_liquid_pairdn_paircoherent_le180_dse_r18_coco_full_'
    '1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
