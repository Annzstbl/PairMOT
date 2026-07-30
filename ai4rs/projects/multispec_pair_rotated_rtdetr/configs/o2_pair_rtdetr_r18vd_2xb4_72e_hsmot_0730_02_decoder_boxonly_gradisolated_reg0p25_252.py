"""0730_02: box-only, gradient-isolated decoder residual (scale 0.25).

This is a strict successor to the formal 0727_01 Liquid+Encoder model.
Classification remains on the shared encoder/decoder representation.  Only
the two frame-specific box refinement paths receive the new zero-start
residual, whose input is detached to avoid an extra gradient route into the
shared classification representation.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_dualoutputresidual_pairdn_paircoherent_le180_coco_full_1200x900_bf16_252 import *  # noqa: F401,F403


model['decoder'].update(
    dual_output_cls_scale=0.0,
    dual_output_reg_scale=0.25,
    dual_output_detach_adapter_input=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0730_02_decoder_boxonly_gradisolated_reg0p25_'
    'pairdn_paircoherent_2xb4_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
