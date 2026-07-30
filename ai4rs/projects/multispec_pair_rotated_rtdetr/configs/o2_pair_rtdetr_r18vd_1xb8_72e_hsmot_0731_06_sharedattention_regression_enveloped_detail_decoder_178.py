"""0731_06: shared classification path plus regression-only frame detail.

The shared-attention decoder remains the source of classification features.
For iterative box refinement only, a zero-start, swap-odd frame correction is
bounded by the raw cross-attention difference and applied as ``-detail`` and
``+detail`` to the two regression heads.  This isolates geometric refinement
from classification after 0731_01 improved detection/AP but slightly reduced
Cls HOTA at epoch 8.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178 import *  # noqa: F401,F403


model['decoder'].update(
    tristate_decoder=False,
    tristate_separate_ffn=False,
    tristate_zero_init_coupling=False,
    dual_output_adapter=False,
    common_motion_decoder=False,
    shared_evidence_decoder=False,
    competitive_evidence_decoder=False,
    motion_trust_decoder=False,
    symmetric_pair_decoder=False,
    shared_routing_decoder=False,
    shared_attention_decoder=True,
    antisymmetric_detail_decoder=False,
    enveloped_detail_decoder=False,
    regression_enveloped_detail_decoder=True,
    common_evidence_bypass_decoder=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0731_06_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_sharedattention_regressionenvelopeddetail_pairdn_paircoherent_'
    'le180_r18_coco_full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
