"""252 continuation port for the mature 0804_01 product tangent.

The scientific model is imported directly from the active 178 1x8 config.
Only the physical realization changes to two ranks with four samples each,
plus the 252 data/cache/evaluation paths.  The global batch, optimizer,
schedulers, model state, decoder geometry, DN, losses, and hooks are unchanged.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0804_01_iterative_cls_terminal_transport_product_tangent_decoder_178 import *  # noqa: F401,F403


# Preserve global batch eight while using fixed 252 GPU0/1.
train_dataloader.update(
    batch_size=4,
    num_workers=2,
    persistent_workers=False)

_hsmot_root = '/data/users/litianhao01/PairMmot/data/hsmot'
_gmc_root = '/data/users/litianhao01/PairMmot/workdir/aux/gmc_cache'
train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1')
val_dataloader.update(num_workers=8, persistent_workers=False)
val_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/test',
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1')
test_dataloader = val_dataloader

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0804_01_terminal_transport_product_tangent_resume252_from_epoch12')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao01/PairMmot/TrackEval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
