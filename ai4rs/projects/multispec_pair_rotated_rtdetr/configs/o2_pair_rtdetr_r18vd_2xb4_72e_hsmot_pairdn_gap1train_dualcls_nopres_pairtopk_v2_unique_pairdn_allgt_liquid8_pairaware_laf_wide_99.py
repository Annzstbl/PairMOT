"""0713_05 liquid8 wide LAF with pair-aware liquid fusion on local 99.

This keeps the current wide liquid-aware fusion recipe and adds a lightweight
pair-aware SE-logit residual.  No band-attention is used: the new branch only
compares compact prev/curr liquid group descriptors after the original sampler
has processed each frame.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_overlap_252 import *  # noqa: F401,F403

_pairmot_root = '/data/users/wangying01/lth/PairMOT'
_gmc_root = f'{_pairmot_root}/workdir/aux/gmc_cache'

model['backbone']['liquid_sampler'].update(
    pair_aware_liquid_fusion=dict(
        hidden_dims=32,
        init_std=1e-3,
        zero_init=True,
    ))

custom_keys['backbone.stem.0.pair_aware_liquid_fusion'] = dict(lr_mult=1.0)

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
    '0714_01_fresh_novis_trackeval_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'liquid8_pairaware_laf_wide')

default_hooks['visualization']['draw'] = False

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_pairmot_root}/data/hsmot/test',
    trackeval_root=f'{_pairmot_root}/TrackEval',
)
test_evaluator = val_evaluator
