"""Four-iteration two-GPU real-data smoke for 0801_11."""
from mmengine.config import read_base

with read_base():
    from ..o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0801_11_terminal_pair_common_cls_residual_decoder_99 import *  # noqa: F401,F403


train_dataloader['batch_size'] = 4
train_cfg = dict(type='IterBasedTrainLoop', max_iters=4, val_interval=1000)
val_cfg = None
val_dataloader = None
val_evaluator = None
default_hooks['checkpoint'].update(
    by_epoch=False, interval=4, max_keep_ckpts=1,
    filename_tmpl='iter_{}.pth')
custom_hooks = [
    hook for hook in custom_hooks
    if hook.get('type') != 'AsyncPairTrackEvalHook'
]
work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    'smoke_0801_11_terminal_pair_common_cls_residual_decoder_4iter')

