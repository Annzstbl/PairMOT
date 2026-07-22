"""Single-GPU accumulation profile with effective batch size eight."""
from mmengine.config import read_base

with read_base():
    from .tmp_profile_0719_pairconsensus_pacde_178_bs8 import *  # noqa: F401,F403


work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    'tmp_profile_0719_pairconsensus_pacde_bs4_acc2')
train_dataloader['batch_size'] = 4
optim_wrapper['accumulative_counts'] = 2
