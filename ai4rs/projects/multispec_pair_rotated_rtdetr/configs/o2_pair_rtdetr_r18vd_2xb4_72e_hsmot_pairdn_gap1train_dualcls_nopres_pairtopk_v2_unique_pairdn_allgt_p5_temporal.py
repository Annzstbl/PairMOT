"""0704_02: 0704_01 + P5 pair temporal adapter before RT-DETR FPN/CCFF.

The adapter runs after the shared AIFI encoder on P5 and before the original
FPN/CCFF.  Its residual scale is zero-initialized, so the initial forward path
is equivalent to 0704_01.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt import *  # noqa: F401,F403

model.encoder.update(
    pair_temporal_adapter_cfg=dict(
        num_heads=4,
        dropout=0.0,
        gamma_init=0.0,
    ),
    pair_temporal_adapter_idx=-1,
)

custom_hooks.append(dict(type='PairTemporalAdapterMonitorHook', interval=50))

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    '0704_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_p5temporal')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
