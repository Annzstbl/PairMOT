from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r18vd_2xb4_72e_hsmot import *

train_dataloader.batch_size = 8

# base_batch_size = (2 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(enable=True, base_batch_size=16)
