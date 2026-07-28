"""Four-iteration DDP smoke for the 0729_02 residual decoder."""
from mmengine.config import read_base

with read_base():
    from ..o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_dualoutputresidual_pairdn_easyhardpositive_le180_coco_full_1200x900_bf16_252 import *  # noqa: F401,F403


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
    '/data4/litianhao/PairMmot/workdir_252/'
    'smoke_0729_02_dualevidence_decoder_dualoutputresidual_'
    'easyhardpositive_4iter')
