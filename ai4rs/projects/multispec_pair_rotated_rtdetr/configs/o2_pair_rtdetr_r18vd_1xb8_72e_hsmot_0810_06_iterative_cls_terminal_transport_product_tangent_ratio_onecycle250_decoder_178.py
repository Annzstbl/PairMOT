"""0810_06: ratio-preserving standard two-phase One-Cycle on the final decoder.

The final terminal-only product-tangent model, global batch, optimizer,
paramwise LR multipliers, EMA, data, losses, hooks, and inference graph remain
unchanged.  ``RatioPreservingOneCycleLR`` only expands the scalar 2.5x peak
factor into MMEngine OneCycleLR's supported per-group ``eta_max`` list.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0804_01_iterative_cls_terminal_transport_product_tangent_decoder_178 import *  # noqa: F401,F403


param_scheduler = [
    dict(
        type='RatioPreservingOneCycleLR',
        eta_max_factor=2.5,
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
    '0810_06_final_product_tangent_ratio_onecycle250_72e_1xb8_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
