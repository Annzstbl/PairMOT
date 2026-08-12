"""0813_01: standard WSD warmup4 + stable56 + cosine12 with 25% floor.

Relative to 0812_01, only the mature cosine schedule floor changes. Model,
data, losses, EMA, optimizer groups, global batch and inference are unchanged.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0810_09_iterative_cls_terminal_transport_product_tangent_wsd4_56_cos12_decoder_252 import *  # noqa: F401,F403


param_scheduler[1]['eta_min_ratio'] = 0.25

_hsmot_root = '/data/users/litianhao/PairMOT/data/hsmot'
_gmc_root = '/data/users/litianhao/PairMOT/workdir/aux/gmc_cache'
train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1')
val_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/test',
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1')
test_dataloader = val_dataloader

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0813_01_final_product_tangent_wsd4_56_cos12_floor25_72e_2xb4_fresh_v2')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao/PairMmot/TrackEval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
