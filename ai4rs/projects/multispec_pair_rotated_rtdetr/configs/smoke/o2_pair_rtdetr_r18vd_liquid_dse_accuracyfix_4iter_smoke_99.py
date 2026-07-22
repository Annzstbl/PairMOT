"""Four-iteration DDP smoke for 0721_05 DSE on server 99."""
from mmengine.config import read_base

with read_base():
    from ..o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_dse_diffproduct_accuracyfix_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


work_dir = '/data4/litianhao/PairMmot/workdir_99/smoke_0721_05_dse_4iter'
train_dataloader['dataset']['indices'] = 32
train_cfg.update(max_epochs=1, val_interval=999)
default_hooks['checkpoint'].update(interval=1, max_keep_ckpts=1)
val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None
