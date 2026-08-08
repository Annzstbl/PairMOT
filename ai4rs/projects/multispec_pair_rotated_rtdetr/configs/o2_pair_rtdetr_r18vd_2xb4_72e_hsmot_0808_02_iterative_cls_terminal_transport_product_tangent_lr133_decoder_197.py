"""0808_02: 96-to-72 integrated-step LR compression on server 197.

The final product-tangent decoder and the full training protocol are retained.
Only AdamW base LR is multiplied by 96/72 = 4/3.  This is a single-factor
test of whether the late e72->e96 improvement is primarily optimizer-clock
limited.  It changes neither parameters nor inference computation.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_26_iterative_cls_terminal_transport_product_tangent_decoder_99 import *  # noqa: F401,F403


optim_wrapper['optimizer']['lr'] = 1.3333333333333334e-4

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
    '0808_02_final_product_tangent_lr133_72e_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao/PairMOT/TrackEval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
