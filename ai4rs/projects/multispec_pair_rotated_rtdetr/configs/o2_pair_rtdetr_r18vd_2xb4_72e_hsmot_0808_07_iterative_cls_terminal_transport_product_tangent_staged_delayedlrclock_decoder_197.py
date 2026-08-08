"""0808_07: staged association-preserving LR clock compression.

The final product-tangent inference model and the complete training protocol
are unchanged. Epochs 1--12 retain the parent's 1e-4 LR. LR is multiplied by
g at epochs 12 and 24, where g = (sqrt(113) - 1) / 8. Hence

    12 * 1 + 12 * g + 48 * g**2 = 96,

so the epoch-72 nominal LR integral exactly matches 96 parent epochs, while
the first transition is only 1.2038x and the final LR remains 1.4491e-4.
This isolates whether a staged clock preserves association better than the
single 1.4x transition in 0808_06. It is class agnostic, uses no reweighting,
and adds no parameter, state, loss, hook, or inference operation.
"""
from mmengine.config import read_base
from mmengine.optim.scheduler.lr_scheduler import MultiStepLR

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0808_02_iterative_cls_terminal_transport_product_tangent_lr133_decoder_197 import *  # noqa: F401,F403


optim_wrapper['optimizer']['lr'] = 1e-4
param_scheduler.append(
    dict(
        type=MultiStepLR,
        begin=0,
        end=72,
        by_epoch=True,
        milestones=[12, 24],
        gamma=1.2037682265918312))

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0808_07_final_product_tangent_staged_delayedlrclock_72e_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
