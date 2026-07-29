"""0729_06: strict old-DN dual-output decoder with cls residual scale 1.25.

This keeps the 0718_01 Liquid base, the 0727_01 Dual-Evidence encoder, the
pair-coherent positive/negative DN protocol, and the dual-output decoder used
by 0729_03.  Regression/reference updates retain residual scale 1.0, while
classification hidden states use the diagnostic Pareto point 1.25.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_decoder0708_03_pairdn_paircoherent_le180_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


model['decoder'].update(
    tristate_decoder=False,
    dual_output_adapter=True,
    dual_output_cls_scale=1.25)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0729_06_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_dualoutputresidual_cls1p25_pairdn_paircoherent_le180_'
    'r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
