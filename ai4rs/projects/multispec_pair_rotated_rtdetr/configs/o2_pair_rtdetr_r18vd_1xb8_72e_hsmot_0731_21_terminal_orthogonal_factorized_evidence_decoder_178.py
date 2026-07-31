"""0731_21: non-shared-attention orthogonal terminal evidence.

This is the missing cell in the structural 2x2 comparison against 0731_18,
0731_19, and 0731_20.  Final classification receives only swap-invariant
common evidence.  Box regression stays on the parent representation and
receives only a strictly antisymmetric 5D detail residual.  Decoder attention
remains independent between frames.
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
    terminal_common_evidence_bypass_decoder=False,
    terminal_classification_common_evidence_decoder=False,
    terminal_factorized_evidence_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0731_21_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminalorthogonalfactorizedevidence_pairdn_paircoherent_le180_'
    'r18_coco_full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
