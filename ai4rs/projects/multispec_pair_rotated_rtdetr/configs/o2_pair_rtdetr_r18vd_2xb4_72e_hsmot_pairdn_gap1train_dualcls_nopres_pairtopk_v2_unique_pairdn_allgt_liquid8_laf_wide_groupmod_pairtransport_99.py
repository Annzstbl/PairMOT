"""0715_02 pair-consistent spectral transport on the 0711_01 baseline.

The two frames retain independent liquid selections.  A bidirectional router
conditions each frame's sampler logits on its pair, then wide LAF transports
cross-frame group tokens according to overlap between the resulting spectral
coverage distributions.  Both pair additions are zero-initialized.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_groupmod_99 import *  # noqa: F401,F403

model['backbone']['liquid_sampler'].update(
    pair_sampler_router=dict(
        hidden_dims=64,
        init_std=1e-3,
        zero_init=True,
    ))
model['backbone']['liquid_sampler']['liquid_aware_fusion'].update(
    pair_transport=dict(
        hidden_dims=128,
        temperature=0.25,
        init_std=1e-3,
        zero_init=True,
    ))

custom_keys['backbone.stem.0.liquid_sampler.pair_sampler_router'] = dict(
    lr_mult=1.0)
custom_keys['backbone.stem.0.liquid_aware_fusion.pair_transport'] = dict(
    lr_mult=1.0)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0715_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'liquid8_laf_wide_groupmod_pairtransport')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
