"""0810_01: migrate the validated 0808_07 epoch-68 state to server 99.

This config changes only host-local data, GMC, TrackEval, and output paths.
The product-tangent model, 2x4 optimization protocol, staged LR scheduler,
EMA clock, losses, and 72-epoch horizon remain exactly those of 0808_07.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0808_07_iterative_cls_terminal_transport_product_tangent_staged_delayedlrclock_decoder_197 import *  # noqa: F401,F403


_hsmot_root = '/data/users/wangying01/lth/PairMOT/data/hsmot'
_gmc_root = '/data/users/wangying01/lth/PairMOT/workdir/aux/gmc_cache'
train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1')
val_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/test',
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1')
test_dataloader = val_dataloader

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0810_01_final_product_tangent_staged_delayedlrclock_'
    'resume_e68_to_e72_2xb4')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/wangying01/lth/PairMOT/TrackEval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
