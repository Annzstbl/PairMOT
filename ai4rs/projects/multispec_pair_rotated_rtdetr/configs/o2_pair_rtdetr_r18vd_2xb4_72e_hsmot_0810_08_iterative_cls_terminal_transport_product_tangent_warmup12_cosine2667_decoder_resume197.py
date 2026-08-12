"""197 host migration of 0810_08 for the exact epoch-68 to epoch-72 resume."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0810_08_iterative_cls_terminal_transport_product_tangent_warmup12_cosine2667_decoder_99 import *  # noqa: F401,F403


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
    '0810_08_final_product_tangent_warmup12_cosine2667_72e_2xb4_'
    'resume_e68_to_e72_197_v2')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao/PairMOT/TrackEval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
