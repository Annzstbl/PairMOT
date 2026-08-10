"""0810_07: ratio-preserving standard two-phase One-Cycle on the final decoder.

This is the 2.0x peak-factor counterpart to 0810_06.  It preserves the final
model and all optimizer paramwise LR ratios and only changes the scheduler.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0804_01_iterative_cls_terminal_transport_product_tangent_decoder_252 import *  # noqa: F401,F403


param_scheduler = [
    dict(
        type='RatioPreservingOneCycleLR',
        eta_max_factor=2.0,
        total_steps=72 * 1038,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1.0e4,
        three_phase=False,
        by_epoch=False)
]

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0810_07_final_product_tangent_ratio_onecycle200_72e_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
