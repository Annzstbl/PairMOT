"""0711_01 liquid8 wide LAF with coverage-aware group modulation on 99.

This combines the two useful signals observed so far: wide overlap LAF gives
the strongest cls/det HOTA on 252, while group modulation improves det-side
stability on 99.  The model keeps pattern-aware SE fusion and additionally
reweights each liquid group from sampling coverage descriptors before SE.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_liquidawarefusion import *  # noqa: F401,F403

model['backbone']['liquid_sampler']['liquid_aware_fusion'].update(
    embed_dims=64,
    num_heads=4,
    use_overlap_context=True,
    use_spatial_mixer=True,
)
model['backbone']['liquid_sampler'].update(
    liquid_group_modulator=dict(
        hidden_dims=16,
        init_std=1e-3,
    ))

custom_keys['backbone.stem.0.liquid_group_modulator'] = dict(lr_mult=1.0)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0711_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'liquid8_laf_wide_groupmod')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
