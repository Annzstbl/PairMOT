"""0812_03: standard warmup4 + cosine68 with a 50% LR floor.

The terminal-only product-tangent model, parameters, data, losses, EMA,
global batch, optimizer groups, and inference graph are unchanged.  The only
scientific change from the zero-floor cosine control is the standard cosine
floor.  The peak is reduced so the nominal 72-epoch LR integral remains 0.0096.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0811_01_iterative_cls_terminal_transport_product_tangent_warmup4_cosine2667_decoder_99 import *  # noqa: F401,F403


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
    '/data4/litianhao/PairMmot/workdir_99/'
    '0812_03_final_product_tangent_warmup4_cosine_floor50_72e_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
