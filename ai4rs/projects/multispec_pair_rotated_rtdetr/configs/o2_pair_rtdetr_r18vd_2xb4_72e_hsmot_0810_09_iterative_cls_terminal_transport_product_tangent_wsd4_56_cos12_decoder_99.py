"""0810_09: standard warmup-stable-decay schedule for the final decoder.

The terminal-only product-tangent model, data, losses, EMA, global batch,
parameter-group multipliers, and inference graph are unchanged.  The sole
scientific change is a conventional WSD learning-rate curve:

* 4 epochs of linear warmup from the inherited 1e-7 safety LR to 1.5e-4;
* 56 epochs at the 1.5e-4 peak;
* 12 epochs of cosine decay to a 1e-4 relative floor.

Ignoring the negligible cosine floor, its nominal LR integral is

    0.5 * 4 * 1.5 + 56 * 1.5 + 0.5 * 12 * 1.5 = 96

parent epochs.  This preserves the e96 update budget without the hand-written
LR jumps used by the successful staged reference.  The schedule is class
agnostic and adds no model parameter, loss, reweighting, or inference work.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_26_iterative_cls_terminal_transport_product_tangent_decoder_99 import *  # noqa: F401,F403


_peak_lr = 1.5e-4
optim_wrapper['optimizer']['lr'] = _peak_lr
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1500.0,
        end_factor=1.0,
        begin=0,
        end=4,
        by_epoch=True),
    dict(
        type='CosineAnnealingLR',
        T_max=12,
        eta_min_ratio=1.0e-4,
        begin=60,
        end=72,
        by_epoch=True),
]

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0810_09_final_product_tangent_wsd4_56_cos12_72e_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
