"""Four-iteration 197 DDP smoke for 0804_03 terminal log-size."""
from mmengine.config import read_base

with read_base():
    from ..o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0804_03_iterative_cls_terminal_log_size_decoder_197 import *  # noqa: F401,F403


train_cfg = dict(type='IterBasedTrainLoop', max_iters=4, val_interval=1000)
val_cfg = None
val_dataloader = None
val_evaluator = None
default_hooks['checkpoint'].update(
    by_epoch=False, interval=4, max_keep_ckpts=1,
    filename_tmpl='iter_{}.pth')
default_hooks['logger'].update(interval=1)
custom_hooks = [
    hook for hook in custom_hooks
    if hook.get('type') != 'AsyncPairTrackEvalHook'
]
work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    'smoke_0804_03_iterative_cls_terminal_log_size_4iter')
