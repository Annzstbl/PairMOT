"""0810_05: conservative standard One-Cycle LR replication on GPU0/1.

The final terminal-only product-tangent model and every non-LR training and
inference setting are inherited unchanged.  Relative to 0810_04, this strict
2x4 realization changes only the One-Cycle peak from 2.5e-4 to 2.0e-4.  It is
the predeclared lower-exposure point, not an unrestricted LR sweep.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0804_01_iterative_cls_terminal_transport_product_tangent_decoder_resume252 import *  # noqa: F401,F403


param_scheduler = [
    dict(
        type='OneCycleLR',
        eta_max=2.0e-4,
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
    '0810_05_final_product_tangent_onecycle200_72e_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
