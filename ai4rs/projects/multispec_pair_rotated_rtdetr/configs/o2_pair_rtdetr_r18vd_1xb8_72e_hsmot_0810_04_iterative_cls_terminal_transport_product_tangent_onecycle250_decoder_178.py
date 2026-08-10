"""0810_04: standard two-phase One-Cycle LR on the final decoder.

This is an optimization-only replacement for the hand-built staged LR clock.
The final terminal-only product-tangent model, global batch, optimizer, EMA,
data, losses, hooks, and inference graph remain unchanged.  The complete
72-epoch schedule uses MMEngine's standard two-phase OneCycleLR with cosine
annealing and its established default phase/division factors.  The 2.5e-4
peak retains approximately the optimizer exposure of the successful long
trajectory while annealing the final updates instead of holding a high LR.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0804_01_iterative_cls_terminal_transport_product_tangent_decoder_178 import *  # noqa: F401,F403


param_scheduler = [
    dict(
        type='OneCycleLR',
        eta_max=2.5e-4,
        total_steps=72 * 1038,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1.0e4,
        three_phase=False,
        by_epoch=False)
]

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0810_04_final_product_tangent_onecycle250_72e_1xb8_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
