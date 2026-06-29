"""Low-LR load-from continuation from formal epoch 72 to 84.

This is a diagnostic run, not a strict resume: optimizer, scheduler, and EMA
state are reset.  It keeps the same model/data/PairDN setup as the formal run
but uses a fixed 2e-5 LR to compare against strict resume at 1e-4.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn import *

load_from = (
    '/data/users/litianhao01/PairMmot/workdir/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn/epoch_72.pth')

optim_wrapper.optimizer.lr = 2e-5
param_scheduler = []

max_epochs = 12
train_cfg.update(max_epochs=max_epochs, val_interval=6)
default_hooks.checkpoint.update(interval=6, max_keep_ckpts=4)

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'loadfrom_epoch72_to84_lr2e5_nosched')
