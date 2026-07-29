"""Single-GPU recovery config for the interrupted 197-side 0729_06 run.

This preserves the global batch size (2x4 -> 1x8), model, losses, data
protocol, and optimizer state.  It is only intended to resume the shared
epoch-4 checkpoint if server 197 remains unavailable after 0729_07 step 7.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0729_06_dualoutput_cls1p25_197 import *  # noqa: F401,F403


_pairmot_root = '/data1/users/litianhao01/PairMOT'
_hsmot_root = f'{_pairmot_root}/data/hsmot'
_gmc_root = f'{_pairmot_root}/workdir/aux/gmc_cache'

train_dataloader['batch_size'] = 8
train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1')
val_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/test',
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1')
test_dataloader = val_dataloader

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0729_06_recovery_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_dualoutputresidual_cls1p25_pairdn_paircoherent_le180_'
    'r18_coco_full_1200x900_bf16_1xb8_from_epoch4')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
