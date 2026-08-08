"""0808_01: conservative global-LR acceleration of the final decoder.

The model, data, losses, global batch, warmup, Liquid schedule, EMA, and
72-epoch endpoint are identical to the final product-tangent decoder.  The
only change is AdamW base LR 1e-4 -> 1.25e-4, testing whether the observed
e72->e96 improvement can be shifted earlier without changing inference.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_26_iterative_cls_terminal_transport_product_tangent_decoder_99 import *  # noqa: F401,F403


optim_wrapper['optimizer']['lr'] = 1.25e-4

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0808_01_final_product_tangent_lr125_72e_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
