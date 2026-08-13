"""0813_02: standard warmup4 + cosine68 with a 65% LR floor.

The model and training protocol are unchanged from 0812_03.  This single
factor raises the standard cosine tail floor while reducing its peak so the
nominal 72-epoch LR integral remains matched.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0812_03_iterative_cls_terminal_transport_product_tangent_warmup4_cosine_floor50_decoder_99 import *  # noqa: F401,F403


_peak_lr = 1.6523235800344234e-4
optim_wrapper['optimizer']['lr'] = _peak_lr
param_scheduler[0]['start_factor'] = 1.0e-7 / _peak_lr
param_scheduler[1]['eta_min_ratio'] = 0.65

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0813_02_final_product_tangent_warmup4_cosine_floor65_72e_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
