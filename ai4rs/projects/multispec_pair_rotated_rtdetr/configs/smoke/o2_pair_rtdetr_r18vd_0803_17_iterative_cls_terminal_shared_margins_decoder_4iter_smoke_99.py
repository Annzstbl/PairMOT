"""Four-iteration 99 DDP smoke for 0803_17."""
from mmengine.config import read_base

with read_base():
    from ..o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_17_iterative_cls_terminal_shared_margins_decoder_99 import *  # noqa: F401,F403


train_cfg = dict(type='IterBasedTrainLoop', max_iters=4, val_interval=1000)
val_cfg = None
val_dataloader = None
val_evaluator = None
default_hooks['checkpoint'].update(
    by_epoch=False, interval=4, max_keep_ckpts=1,
    filename_tmpl='iter_{}.pth')
default_hooks['logger'].update(interval=1)
custom_hooks = [h for h in custom_hooks
                if h.get('type') != 'AsyncPairTrackEvalHook']
work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    'smoke_0803_17_iterative_cls_terminal_shared_margins_4iter_99')
