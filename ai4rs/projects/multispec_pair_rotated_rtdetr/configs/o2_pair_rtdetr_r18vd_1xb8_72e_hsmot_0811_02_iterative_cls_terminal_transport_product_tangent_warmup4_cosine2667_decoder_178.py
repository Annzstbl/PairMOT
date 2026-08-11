"""0811_02: standard four-epoch warmup plus cosine on one 178 GPU.

This is the 1x8 resource-equivalent form of 0811_01.  The terminal-only
product-tangent model, data, losses, EMA, global batch, optimizer parameter
groups, and inference graph are unchanged.  Only the standard scheduler uses
four warmup epochs to the same 8/3 parent peak, followed by 68 cosine epochs.
The nominal LR integral remains 96 parent epochs.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0804_01_iterative_cls_terminal_transport_product_tangent_decoder_178 import *  # noqa: F401,F403


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
    '/data4/litianhao/PairMmot/workdir_178/'
    '0811_02_final_product_tangent_warmup4_cosine2667_72e_1xb8_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
