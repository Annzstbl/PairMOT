"""0810_08: standard 12-epoch linear-warmup cosine schedule.

The final terminal-only product-tangent model, data, losses, EMA, global
batch, parameter-group multipliers, and inference graph are unchanged.  The
only scientific change is a conventional linear-warmup plus cosine LR curve.
The warmup reaches 8/3 times the parent's base LR at epoch 12 and the final
60 epochs use cosine annealing.  Its nominal LR integral is

    0.5 * 12 * (8/3) + 0.5 * 60 * (8/3) = 96

parent epochs, while the initial LR remains the parent's 1e-7 safety value.
The schedule is class agnostic and adds no parameter or inference operation.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_26_iterative_cls_terminal_transport_product_tangent_decoder_99 import *  # noqa: F401,F403


_peak_lr = 8.0e-4 / 3.0
optim_wrapper['optimizer']['lr'] = _peak_lr
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.000375,
        end_factor=1.0,
        begin=0,
        end=12,
        by_epoch=True),
    dict(
        type='CosineAnnealingLR',
        T_max=60,
        eta_min_ratio=1.0e-4,
        begin=12,
        end=72,
        by_epoch=True),
]

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0810_08_final_product_tangent_warmup12_cosine2667_72e_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
