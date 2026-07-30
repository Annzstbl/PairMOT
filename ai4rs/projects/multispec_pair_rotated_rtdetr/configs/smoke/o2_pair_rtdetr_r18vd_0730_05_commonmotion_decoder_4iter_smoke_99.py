"""Four-iteration real-data DDP smoke for the 0730_05 decoder."""
from mmengine.config import read_base

with read_base():
    from ..o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0730_05_commonmotion_decoder_pairdn_paircoherent_99 import *  # noqa: F401,F403


train_cfg = dict(type='IterBasedTrainLoop', max_iters=4, val_interval=1000)
param_scheduler = []
val_cfg = None
test_cfg = None
val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = None
default_hooks['checkpoint'].update(
    by_epoch=False, interval=4, save_last=True, max_keep_ckpts=1)
default_hooks['logger'].update(interval=1)
custom_hooks = [
    hook for hook in custom_hooks
    if hook.get('type') != 'AsyncPairTrackEvalHook'
]
work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    'smoke_0730_05_commonmotion_decoder_4iter')
