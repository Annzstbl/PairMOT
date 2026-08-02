"""Four-iteration single-GPU real-data smoke for 0803_02 on 178."""
from mmengine.config import read_base

with read_base():
    from ..o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0803_02_iterative_cls_pair_shared_shape_refinement_decoder_178 import *  # noqa: F401,F403


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
    '/data4/litianhao/PairMmot/workdir_178/'
    'smoke_0803_02_iterative_cls_pair_shared_shape_refinement_decoder_4iter')
