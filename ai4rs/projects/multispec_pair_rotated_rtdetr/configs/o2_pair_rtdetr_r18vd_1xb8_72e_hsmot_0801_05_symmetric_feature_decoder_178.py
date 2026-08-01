"""0801_05: feature-only symmetric pair decoder.

Only the per-layer frame-feature fusion into the shared recurrent query is
made invariant to frame order. Cross-attention remains frame-specific and the
original ordered pair-position encoding is retained. Repeating the feature
mean through the existing fusion layer adds no parameters, matrix
multiplications, branches, losses, or decoder depth.
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
    symmetric_position_decoder=False,
    symmetric_feature_decoder=True,
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
    terminal_factorized_evidence_decoder=False,
    terminal_factorized_confidence='none',
    terminal_factorized_diagonal_gates=False,
    terminal_factorized_coupled_gate=False,
    terminal_factorized_center_motion_only=False,
    terminal_factorized_detail_only=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0801_05_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_symmetricfeature_pairdn_paircoherent_le180_r18_'
    'coco_full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
