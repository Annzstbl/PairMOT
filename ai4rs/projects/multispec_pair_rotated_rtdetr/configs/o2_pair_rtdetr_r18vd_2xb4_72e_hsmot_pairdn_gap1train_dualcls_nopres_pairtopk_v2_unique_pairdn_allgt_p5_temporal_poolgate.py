"""0704_04: 0704_01 + lightweight P5 temporal pool-gate adapter.

Compared with full P5 cross-attention, this adapter uses global two-frame
context to gate a small convolutional delta.  The residual scale is initialized
to zero, so the initial forward path is equivalent to 0704_01.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt import *  # noqa: F401,F403

model.encoder.update(
    pair_temporal_adapter_cfg=dict(
        type='pool_gate',
        reduction=4,
        gamma_init=0.0,
    ),
    pair_temporal_adapter_idx=-1,
)

custom_hooks.append(dict(type='PairTemporalAdapterMonitorHook', interval=50))

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    '0704_04_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_p5temporal_poolgate')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
