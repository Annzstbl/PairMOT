"""0710_01 liquid8 with coverage-aware group modulation on 99.

This keeps the best plain liquid8 setting and inserts a lightweight model
branch before SE fusion.  The branch reads each group's sampled band coverage,
coverage entropy, peak coverage and conv3d response, then reweights the group
feature.  It tests whether the strong plain liquid8 result benefits from
coverage-aware group balancing without adding the heavier LAF branch.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8 import *  # noqa: F401,F403

model['backbone']['liquid_sampler'].update(
    liquid_group_modulator=dict(
        hidden_dims=16,
        init_std=1e-3,
    ))

custom_keys['backbone.stem.0.liquid_group_modulator'] = dict(lr_mult=1.0)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0710_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'liquid8_groupmod')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
