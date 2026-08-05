"""Four-iteration fixed-GPU 252 DDP smoke for the 0806_04 continuation."""
from mmengine.config import read_base

with read_base():
    from ..o2_pair_rtdetr_r18vd_2xb4_88e_hsmot_0806_04_iterative_cls_terminal_transport_product_tangent_decoder_resume252 import *  # noqa: F401,F403


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
    '/data4/litianhao/PairMmot/workdir_252/'
    'smoke_0806_04_terminal_transport_product_tangent_4iter')
