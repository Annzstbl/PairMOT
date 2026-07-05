"""0705_01: 0704_02 + lightweight multi-scale local temporal adapter.

Both temporal branches are zero-gated at initialization, so the initial forward
path stays equivalent to the same all-GT unique baseline.  This config does not
load from a trained epoch checkpoint; it uses the same pretrain chain as
0704_01/0704_02 for fair attribution.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_p5_temporal import *  # noqa: F401,F403

model.encoder.update(
    post_pair_temporal_adapter_cfg=dict(
        type='pyramid_local',
        in_channels=[256, 256, 256],
        level_indices=[0, 1, 2],
        reduction=4,
        pointwise_groups=8,
        gamma_init=0.0,
    ),
)

optim_wrapper['paramwise_cfg']['custom_keys'].update({
    'encoder.pair_temporal_adapter.gamma': dict(
        lr_mult=20.0,
        decay_mult=0.0,
    ),
    'encoder.pair_temporal_adapter': dict(
        lr_mult=2.0,
    ),
    'encoder.post_pair_temporal_adapter.gamma': dict(
        lr_mult=20.0,
        decay_mult=0.0,
    ),
    'encoder.post_pair_temporal_adapter': dict(
        lr_mult=2.0,
    ),
})

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    '0705_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'p5temporal_pyramidlocal')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
