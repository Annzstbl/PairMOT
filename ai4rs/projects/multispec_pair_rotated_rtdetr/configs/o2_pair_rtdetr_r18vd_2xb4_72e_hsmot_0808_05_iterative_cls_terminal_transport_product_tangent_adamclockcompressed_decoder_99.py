"""0808_05: complete 96-to-72 optimizer-clock compression.

The final product-tangent inference model is unchanged.  In addition to the
LR, warmup, EMA, and Liquid clocks used by 0808_04, Adam's exponential-memory
clocks are compressed by the same 96/72 factor.  For an original decay beta,
the compressed value is beta ** (96 / 72), so 72 epochs expose the first- and
second-moment estimates to the same nominal decay as 96 parent epochs.

This is class agnostic, uses no reweighting, and adds no model parameter,
state, loss, or inference operation.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_26_iterative_cls_terminal_transport_product_tangent_decoder_99 import *  # noqa: F401,F403


_clock_ratio = 96.0 / 72.0
optim_wrapper['optimizer'].update(
    lr=1.3333333333333334e-4,
    betas=(0.9 ** _clock_ratio, 0.999 ** _clock_ratio))
param_scheduler[0]['end'] = 1500
custom_hooks[3].update(momentum=1.3333333333333334e-4, gamma=1500)
custom_hooks[4].update(anneal_epochs=27, hard_start_epoch=27)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0808_05_final_product_tangent_adamclockcompressed_72e_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
