"""0731_23: gradient-isolated orthogonal terminal evidence.

This is the detached-gate counterpart of independent-attention 0731_21.
Common evidence remains classification-only and frame detail remains a
strictly antisymmetric terminal box residual, while neither gate can add a
gradient path into the frame-evidence tensors.
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
    terminal_factorized_evidence_decoder=True,
    terminal_detach_gate_evidence=True)

work_dir = ('/data4/litianhao/PairMmot/workdir_252/'
            '0731_23_paper_base_liquid_encoder_p5temporal_dualevidence_'
            'decoder_terminaldetachedorthogonalfactorizedevidence_'
            'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao01/PairMmot/TrackEval',
    track_data_root='/data/users/litianhao01/PairMmot/data/hsmot/test')
test_evaluator = val_evaluator
