"""Exact-resolution single-5090 four-iteration smoke for AutoDL 0726_02."""
from mmengine.config import read_base

with read_base():
    from ..autodl_0726_02_paper_base_liquid_encoder_p5temporal_pyramidlocal_pairdn_paircoherent_le180_full_1200x900_bf16_1xb8 import *  # noqa: F401,F403


train_dataloader['dataset']['indices'] = 64
train_cfg = dict(type='IterBasedTrainLoop', max_iters=4, val_interval=1000)
param_scheduler = []
default_hooks['checkpoint'].update(
    by_epoch=False, interval=4, save_last=True, max_keep_ckpts=1)
default_hooks['logger'].update(interval=1)
val_cfg = None
test_cfg = None
val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = None
work_dir = (
    '/root/autodl-tmp/work_dirs/'
    'smoke_0726_02_base_liquid_encoder_pairdn_1xb8_4iter')
