"""Four-iteration real-data smoke for 0808_03 on server 178."""
from mmengine.config import read_base

with read_base():
    from ..o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0808_03_iterative_cls_terminal_transport_product_tangent_decoderheadlr133_decoder_178 import *  # noqa: F401,F403


train_cfg = dict(type='IterBasedTrainLoop', max_iters=4, val_interval=1000)
val_cfg = None
val_dataloader = None
val_evaluator = None
default_hooks['checkpoint'].update(
    by_epoch=False, interval=4, max_keep_ckpts=1, filename_tmpl='iter_{}.pth')
default_hooks['logger'].update(interval=1)
custom_hooks = [
    hook for hook in custom_hooks
    if hook.get('type') != 'AsyncPairTrackEvalHook'
]
work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    'smoke_0808_03_product_tangent_decoderheadlr133_4iter')
