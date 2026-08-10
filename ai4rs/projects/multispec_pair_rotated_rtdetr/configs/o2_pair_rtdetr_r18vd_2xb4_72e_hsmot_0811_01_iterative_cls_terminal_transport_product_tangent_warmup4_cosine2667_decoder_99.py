"""0811_01: standard four-epoch warmup plus cosine schedule.

This is a single-factor successor to 0810_08.  The final terminal-only
product-tangent model, data, losses, EMA, global batch, parameter-group
multipliers, and inference graph are unchanged.  Only the standard scheduler
phase boundary changes: linear warmup reaches the same 8/3 parent peak at
epoch 4, followed by 68 epochs of cosine annealing.  Its nominal LR integral
is unchanged:

    0.5 * 4 * (8/3) + 0.5 * 68 * (8/3) = 96

parent epochs.  The schedule is class agnostic and adds no parameter,
reweighting, loss, or inference operation.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0810_08_iterative_cls_terminal_transport_product_tangent_warmup12_cosine2667_decoder_99 import *  # noqa: F401,F403


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.000375,
        end_factor=1.0,
        begin=0,
        end=4,
        by_epoch=True),
    dict(
        type='CosineAnnealingLR',
        T_max=68,
        eta_min_ratio=1.0e-4,
        begin=4,
        end=72,
        by_epoch=True),
]

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0811_01_final_product_tangent_warmup4_cosine2667_72e_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
