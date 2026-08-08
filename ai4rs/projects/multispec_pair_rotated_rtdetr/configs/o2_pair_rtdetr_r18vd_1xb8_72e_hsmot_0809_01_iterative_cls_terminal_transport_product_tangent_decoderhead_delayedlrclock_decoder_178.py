"""0809_01: delay clock compression to decoder/head parameters only.

The first 12 epochs exactly retain the final product-tangent parent's LR for
every parameter.  At the epoch-12 boundary, only decoder and prediction-head
groups receive a 1.4x LR multiplier.  Their epoch-integrated LR is therefore
12 * 1 + 60 * 1.4 = 96 parent epochs by epoch 72, while backbone and encoder
optimization remain on the validated 72-epoch trajectory.

This is class agnostic, uses no reweighting, and changes no model parameter,
state, loss, or inference operation.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0804_01_iterative_cls_terminal_transport_product_tangent_decoder_178 import *  # noqa: F401,F403


_tag_key = 'decoder_head_delayed_lr'
optim_wrapper['paramwise_cfg']['custom_keys'].update({
    'decoder': dict(lr_mult=1.0, decoder_head_delayed_lr=True),
    'bbox_head': dict(lr_mult=1.0, decoder_head_delayed_lr=True),
})
param_scheduler.append(
    dict(
        type='TaggedMultiStepLR',
        begin=0,
        end=72,
        by_epoch=True,
        milestones=[12],
        gamma=1.4,
        tag_key=_tag_key))

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0809_01_final_product_tangent_decoderhead_delayedlrclock_72e_1xb8_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
