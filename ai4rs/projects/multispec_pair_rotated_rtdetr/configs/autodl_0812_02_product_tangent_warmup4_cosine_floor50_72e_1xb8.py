"""AutoDL fallback: warmup4 plus cosine68 with a 50% LR floor.

The terminal-only product-tangent model, parameters, data, losses, EMA,
physical/global batch, and inference graph are identical to 0811_02.  The
only scientific change is a standard CosineAnnealingLR floor of 0.5.  Its
peak is reduced to preserve the same nominal 96-parent-epoch LR integral:

    peak * (0.5 * 4 + 0.5 * 68 * (1 + 0.5)) = 0.0096,

so peak = 1.811320754716981e-4 and the final floor is
9.056603773584906e-5.  This class-agnostic schedule adds no reweighting,
model state, loss, hook, or inference computation.
"""
from mmengine.config import read_base

with read_base():
    from .autodl_0811_02_product_tangent_warmup4_cosine2667_72e_1xb8 import *  # noqa: F401,F403


_peak_lr = 1.811320754716981e-4
optim_wrapper['optimizer']['lr'] = _peak_lr
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-7 / _peak_lr,
        end_factor=1.0,
        begin=0,
        end=4,
        by_epoch=True),
    dict(
        type='CosineAnnealingLR',
        T_max=68,
        eta_min_ratio=0.5,
        begin=4,
        end=72,
        by_epoch=True),
]

work_dir = (
    '/root/autodl-tmp/work_dirs/'
    '0812_02_final_product_tangent_warmup4_cosine_floor50_72e_1xb8_'
    'autodl_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
