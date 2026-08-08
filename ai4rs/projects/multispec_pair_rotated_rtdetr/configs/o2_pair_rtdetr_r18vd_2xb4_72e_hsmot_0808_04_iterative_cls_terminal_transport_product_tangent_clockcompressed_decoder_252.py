"""0808_04: coherent 96-to-72 optimization-clock compression.

The final product-tangent inference model is unchanged.  Training time is
compressed by 72/96: base LR is multiplied by 4/3, while iteration/epoch
clocks controlling warmup, EMA startup, and Liquid hardening are multiplied
by 3/4.  This preserves the intended ordering of optimization phases while
targeting the e96 maturity level at epoch 72.  No class-aware operation,
reweighting, parameter, state, loss, or inference compute is introduced.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0804_01_iterative_cls_terminal_transport_product_tangent_decoder_resume252 import *  # noqa: F401,F403


optim_wrapper['optimizer']['lr'] = 1.3333333333333334e-4
param_scheduler[0]['end'] = 1500
custom_hooks[3].update(momentum=1.3333333333333334e-4, gamma=1500)
custom_hooks[4].update(anneal_epochs=27, hard_start_epoch=27)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0808_04_final_product_tangent_clockcompressed_72e_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
