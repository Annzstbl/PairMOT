"""Exact 197 resume of standard WSD from the shared 252 epoch-36 state."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0810_09_iterative_cls_terminal_transport_product_tangent_wsd4_56_cos12_decoder_197 import *  # noqa: F401,F403


work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0812_01_final_product_tangent_wsd4_56_cos12_72e_2xb4_resume_e36_to_e72')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator

