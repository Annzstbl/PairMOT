"""0705_03: 0704_01 + post-FPN pyramid-local temporal adapter on P4/P5.

This is the semantic-scale ablation for 0705_02.  It keeps the same all-GT
unique proposal baseline and the same adapted pretrain, but applies the local
temporal adapter only on the two higher FPN levels.  The zero-initialized
per-level residual scale keeps the initial forward path equivalent to 0704_01.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt import *  # noqa: F401,F403

model.encoder.update(
    post_pair_temporal_adapter_cfg=dict(
        type='pyramid_local',
        in_channels=[256, 256, 256],
        level_indices=[1, 2],
        reduction=4,
        pointwise_groups=8,
        gamma_init=0.0,
    ),
)

custom_hooks.append(dict(type='PairTemporalAdapterMonitorHook', interval=50))

optim_wrapper['paramwise_cfg']['custom_keys'].update({
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
    '0705_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'pyramidlocal_p4p5')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
