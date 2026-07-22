"""Data-worker profile only; never use this config for formal training."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_liquid_pairconsensus_reliability_coco_full_1200x900_bf16_178 import *  # noqa: F401,F403


work_dir = '/data4/litianhao/PairMmot/workdir_178/profile_0719_02_workers'
train_dataloader['dataset']['indices'] = 320
train_cfg.update(max_epochs=1, val_interval=999)

val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None

default_hooks['logger']['interval'] = 1
default_hooks['checkpoint'].update(interval=999, save_last=False)
