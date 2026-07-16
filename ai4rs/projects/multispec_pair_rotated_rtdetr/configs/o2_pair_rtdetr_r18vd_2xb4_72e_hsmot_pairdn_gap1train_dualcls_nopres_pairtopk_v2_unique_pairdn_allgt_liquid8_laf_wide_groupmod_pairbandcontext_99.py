"""0715_03 band-aligned pair context on the 0711_01 baseline.

Prev/curr physical bands build a shared directional context.  The same context
updates sampler band descriptors and is pooled by the sampled coverage into
wide-LAF group tokens.  Both injections are zero-initialized.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_groupmod_99 import *  # noqa: F401,F403

model['backbone']['liquid_sampler'].update(
    pair_band_context=dict(
        hidden_dims=64,
        init_std=1e-3,
        zero_init=True,
    ))
model['backbone']['liquid_sampler']['liquid_aware_fusion'].update(
    pair_band_context_fusion=dict(
        context_dims=32,
        hidden_dims=64,
        init_std=1e-3,
        zero_init=True,
    ))

custom_keys['backbone.stem.0.liquid_sampler.pair_band_context'] = dict(
    lr_mult=1.0)
custom_keys[
    'backbone.stem.0.liquid_aware_fusion.pair_band_context_fusion'] = dict(
        lr_mult=1.0)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0715_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'liquid8_laf_wide_groupmod_pairbandcontext')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
