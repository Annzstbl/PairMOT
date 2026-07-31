"""0731_16: terminal-only common-evidence bypass on the 0727_01 parent.

The recurrent decoder query, all auxiliary predictions, and every reference
consumed by a later layer stay on the parent path. Only the final heads receive
a shared, swap-invariant correction toward raw two-frame common evidence.
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
    shared_attention_decoder=False,
    antisymmetric_detail_decoder=False,
    enveloped_detail_decoder=False,
    regression_enveloped_detail_decoder=False,
    midpoint_regression_enveloped_detail_decoder=False,
    classification_enveloped_detail_decoder=False,
    terminal_enveloped_detail_decoder=False,
    terminal_midpoint_enveloped_detail_decoder=False,
    terminal_regression_enveloped_detail_decoder=False,
    terminal_midpoint_regression_enveloped_detail_decoder=False,
    common_evidence_bypass_decoder=False,
    terminal_common_evidence_bypass_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0731_16_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminalcommonevidencebypass_pairdn_paircoherent_le180_'
    'r18_coco_full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
