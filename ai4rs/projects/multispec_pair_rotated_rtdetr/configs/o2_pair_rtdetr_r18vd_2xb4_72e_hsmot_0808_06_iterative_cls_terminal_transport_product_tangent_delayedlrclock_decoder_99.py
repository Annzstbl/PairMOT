"""0808_06: association-preserving delayed LR clock compression.

The final product-tangent inference model is unchanged.  Epochs 1--12 retain
the parent's 1e-4 learning rate so early pair association can form under the
validated optimization dynamics.  At the epoch-12 boundary, LR is multiplied
once by 1.4 for the remaining 60 epochs.  The nominal epoch-integrated LR is
therefore 12 * 1 + 60 * 1.4 = 96 parent epochs by epoch 72.

This is class agnostic, uses no reweighting, and adds no model parameter,
state, loss, hook, or inference operation.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_26_iterative_cls_terminal_transport_product_tangent_decoder_99 import *  # noqa: F401,F403


param_scheduler.append(
    dict(
        type='MultiStepLR',
        begin=0,
        end=72,
        by_epoch=True,
        milestones=[12],
        gamma=1.4))

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0808_06_final_product_tangent_delayedlrclock_72e_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
