"""Four-iteration DDP smoke for the accuracy-fixed 0721_02 rerun."""
from mmengine.config import read_base

with read_base():
    from ..o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_independent_diffproduct_accuracyfix_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    'smoke_0721_02_liquid_independent_diffproduct_accuracyfix_4iter')
train_dataloader['dataset']['indices'] = 32
train_cfg.update(max_epochs=1, val_interval=999)
default_hooks['checkpoint'].update(interval=1, max_keep_ckpts=1)
val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None
