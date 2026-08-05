"""0806_06: optional duration-only continuation from e88 to e96.

This contingency changes only the terminal epoch, evaluation horizon, and
output paths.  It must not be deployed unless the complete epoch-88 audit
still misses one of the three strict final thresholds.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_88e_hsmot_0806_04_iterative_cls_terminal_transport_product_tangent_decoder_resume252 import *  # noqa: F401,F403


max_epochs = 96
train_cfg.update(max_epochs=max_epochs, val_interval=4)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0806_06_terminal_transport_product_tangent_resume252_e88_to_e96')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
