"""0813_03: standard WSD warmup4 + stable44 + cosine24 floor25.

The model and training protocol are unchanged from 0812_04.  The sole factor
is a 25% nonzero floor for the conventional cosine decay; its peak is reduced
to preserve the nominal 72-epoch LR integral.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0812_04_iterative_cls_terminal_transport_product_tangent_wsd4_44_cos24_decoder_178 import *  # noqa: F401,F403


_peak_lr = 1.5737704918032788e-4
optim_wrapper['optimizer']['lr'] = _peak_lr
param_scheduler[0]['start_factor'] = 1.0e-7 / _peak_lr
param_scheduler[1]['eta_min_ratio'] = 0.25

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0813_03_final_product_tangent_wsd4_44_cos24_floor25_72e_1xb8_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
