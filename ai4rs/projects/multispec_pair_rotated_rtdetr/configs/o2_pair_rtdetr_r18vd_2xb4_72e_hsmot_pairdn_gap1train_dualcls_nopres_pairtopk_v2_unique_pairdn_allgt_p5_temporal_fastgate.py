"""0704_03: P5 temporal adapter with a faster residual gate.

This keeps the zero-initialized residual gate from 0704_02, but gives the
temporal adapter a dedicated optimizer rule so the gate can open earlier while
the inherited detector/backbone learning rates remain unchanged.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_p5_temporal import *  # noqa: F401,F403

optim_wrapper['paramwise_cfg']['custom_keys'].update({
    'encoder.pair_temporal_adapter.gamma': dict(
        lr_mult=20.0,
        decay_mult=0.0,
    ),
    'encoder.pair_temporal_adapter': dict(
        lr_mult=2.0,
    ),
})

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    '0704_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_p5temporal_fastgate')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
