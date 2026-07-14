"""0712_01 liquid8 wide LAF + groupmod with a small output residual on 99.

The 0711_01 wide LAF + groupmod run is the current strongest 99 liquid result.
This variant keeps that structure and lets the liquid-aware fusion delta also
inject a small residual into the final stem output, testing whether the
det-side benefit previously seen from output residual can be retained once
group-level modulation stabilizes the liquid features.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_groupmod_99 import *  # noqa: F401,F403

model['backbone']['liquid_sampler']['liquid_aware_fusion'].update(
    output_residual=dict(init_value=0.02),
)

custom_keys['backbone.stem.0.liquid_output_residual_scale'] = dict(
    lr_mult=1.0)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0712_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'liquid8_laf_wide_groupmod_outputres')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
