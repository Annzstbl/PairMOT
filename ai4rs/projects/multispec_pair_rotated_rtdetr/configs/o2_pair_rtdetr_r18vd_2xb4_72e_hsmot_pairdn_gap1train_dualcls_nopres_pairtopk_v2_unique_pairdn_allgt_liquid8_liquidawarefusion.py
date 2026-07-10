"""0709 liquid8 with liquid-aware spectral fusion.

The sampler still starts from 8 cyclic 3-band groups.  The new fusion branch
adds an SE-logit residual from the liquid sampling distribution P and conv3d
group responses, allowing the gate to react to which source bands each group
actually used.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8 import *  # noqa: F401,F403

model['backbone']['liquid_sampler'].update(
    liquid_aware_fusion=dict(
        embed_dims=32,
        num_heads=4,
        spatial_kernel=3,
        dropout=0.0,
        init_std=1e-3,
    ))

custom_keys['backbone.stem.0.liquid_aware_fusion'] = dict(lr_mult=1.0)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0709_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'liquid8_liquidawarefusion')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
