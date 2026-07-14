"""0710_02 liquid8 LAF overlap with pattern-aware output residual on 197.

The previous LAF variants only changed SE logits.  This variant keeps the
best LAF-overlap setting and lets the LAF spatial delta also modulate a small
residual added to the final stem output, so pattern information can affect the
feature map directly instead of only changing group gates.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_overlap_197 import *  # noqa: F401,F403

model['backbone']['liquid_sampler']['liquid_aware_fusion'].update(
    output_residual=dict(init_value=0.05),
)

custom_keys['backbone.stem.0.liquid_output_residual_scale'] = dict(lr_mult=1.0)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0710_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'liquid8_laf_outputres')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
