"""0809_02: compress the EMA clock after delaying local LR acceleration.

This keeps the 0809_01 optimization trajectory: every parameter follows the
validated parent LR through epoch 12, then only decoder and prediction-head
groups receive a 1.4x multiplier.  The sole additional factor is a 96-to-72
compression of the EMA clock.  Its asymptotic momentum is multiplied by 4/3
and its early annealing gamma by 3/4, preserving the parent EMA's nominal
sample-time scale under the shortened schedule.

The model, parameter/state shapes, losses, data, and inference graph are
unchanged.  The strategy is class agnostic and uses no reweighting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_26_iterative_cls_terminal_transport_product_tangent_decoder_99 import *  # noqa: F401,F403


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

# ExpMomentumEMA defaults to momentum=1e-4 and gamma=2000.  Scaling both
# clocks by 96/72 makes the EMA at epoch 72 use the same nominal sample-time
# horizon as the parent protocol at epoch 96.
custom_hooks[3].update(
    momentum=1.3333333333333334e-4,
    gamma=1500)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0809_02_final_product_tangent_decoderhead_delayedlrclock_'
    'emaclock_72e_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
