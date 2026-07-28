"""Four-iteration real-data DDP smoke for 0725_01."""
from mmengine.config import read_base

with read_base():
    from ..o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_pairdn_paircoherent_le180_dse_cpdse_pairglobal_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


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
    '/data4/litianhao/PairMmot/workdir_197/'
    'smoke_0725_01_pairdn_dse_cpdse_pairglobal_4iter_retry2')
