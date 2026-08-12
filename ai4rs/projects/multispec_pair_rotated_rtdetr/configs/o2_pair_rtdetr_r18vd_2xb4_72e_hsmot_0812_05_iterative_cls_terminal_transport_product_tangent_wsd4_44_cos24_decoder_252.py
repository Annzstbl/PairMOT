"""0812_05: standard WSD warmup4 + stable44 + cosine24 on 252.

This is the 2x4 realization of the mature 0812_04 schedule.  Model, data,
losses, EMA, optimizer groups, global batch and inference are unchanged.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0810_09_iterative_cls_terminal_transport_product_tangent_wsd4_56_cos12_decoder_252 import *  # noqa: F401,F403


_peak_lr = 1.6551724137931035e-4
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
        T_max=24,
        eta_min_ratio=1.0e-4,
        begin=48,
        end=72,
        by_epoch=True),
]

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0812_05_final_product_tangent_wsd4_44_cos24_72e_2xb4_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
