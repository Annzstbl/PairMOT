"""0813_01: standard WSD warmup4 + stable56 + cosine12 with 25% floor.

Relative to 0812_01, only the mature cosine schedule floor changes. Model,
data, losses, EMA, optimizer groups, global batch and inference are unchanged.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0810_09_iterative_cls_terminal_transport_product_tangent_wsd4_56_cos12_decoder_252 import *  # noqa: F401,F403


param_scheduler[1]['eta_min_ratio'] = 0.25

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0813_01_final_product_tangent_wsd4_56_cos12_floor25_72e_2xb4_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
