"""0730_03: box-only, gradient-isolated decoder residual (scale 0.5)."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_dualoutputresidual_pairdn_paircoherent_le180_coco_full_1200x900_bf16_252 import *  # noqa: F401,F403


model['decoder'].update(
    dual_output_cls_scale=0.0,
    dual_output_reg_scale=0.5,
    dual_output_detach_adapter_input=True)
