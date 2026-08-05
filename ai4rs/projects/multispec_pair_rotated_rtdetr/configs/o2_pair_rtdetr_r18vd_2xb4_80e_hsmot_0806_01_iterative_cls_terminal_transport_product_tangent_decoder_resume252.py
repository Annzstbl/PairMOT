"""0806_01: extend the mature 0804_01 product-tangent run to epoch 80.

This is a duration-only continuation from the audited epoch-72 checkpoint.
The model, global batch, optimizer, warmup state, EMA, losses, data, decoder
geometry, and evaluation cadence remain identical to the 252 port.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0804_01_iterative_cls_terminal_transport_product_tangent_decoder_resume252 import *  # noqa: F401,F403


max_epochs = 80
train_cfg.update(max_epochs=max_epochs, val_interval=4)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0806_01_terminal_transport_product_tangent_resume252_e72_to_e80')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
