"""Four real-data DDP iterations resumed exactly after WSD epoch 36."""
from mmengine.config import read_base

with read_base():
    from ..o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0812_01_iterative_cls_terminal_transport_product_tangent_wsd4_56_cos12_decoder_resume197 import *  # noqa: F401,F403


train_cfg = dict(type='IterBasedTrainLoop', max_iters=37372, val_interval=100000)
val_cfg = None
val_dataloader = None
val_evaluator = None
default_hooks['checkpoint'].update(
    by_epoch=False, interval=37372, max_keep_ckpts=1,
    filename_tmpl='iter_{}.pth')
default_hooks['logger'].update(interval=1)
work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    'smoke_0812_01_product_tangent_wsd_resume_e36_4iter')

