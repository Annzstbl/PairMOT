"""178 single-GPU 0730_04 box-only decoder residual experiment."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178 import *  # noqa: F401,F403


model['decoder'].update(
    tristate_decoder=False,
    dual_output_adapter=True,
    dual_output_cls_scale=0.0,
    dual_output_reg_scale=1.0,
    dual_output_detach_adapter_input=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0730_04_decoder_boxonly_gradisolated_reg1p0_'
    'pairdn_paircoherent_1xb8_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
