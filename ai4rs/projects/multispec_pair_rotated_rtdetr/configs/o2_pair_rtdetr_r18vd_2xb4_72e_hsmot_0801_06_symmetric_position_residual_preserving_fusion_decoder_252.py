"""0801_06: position symmetry plus residual-preserving frame fusion.

Position-only symmetry gives a positive Det HOTA/DetA signal at epoch 4 but
slightly lowers Cls DetA. This minimal successor keeps that position path and
makes every recurrent update preserve the post-self-attention shared query
explicitly. The existing fusion learns only the two frame cross-attention
innovations. It adds no parameters, matrix multiplications, decoder layers,
attention modules, branches, losses, or tunable scales.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_dualoutputresidual_pairdn_paircoherent_le180_coco_full_1200x900_bf16_252 import *  # noqa: F401,F403


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
    symmetric_position_decoder=True,
    symmetric_feature_decoder=False,
    residual_preserving_fusion_decoder=True,
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
    '/data4/litianhao/PairMmot/workdir_252/'
    '0801_06_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_symmetricposition_residualpreservingfusion_pairdn_'
    'paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao01/PairMmot/TrackEval',
    track_data_root='/data/users/litianhao01/PairMmot/data/hsmot/test')
test_evaluator = val_evaluator
