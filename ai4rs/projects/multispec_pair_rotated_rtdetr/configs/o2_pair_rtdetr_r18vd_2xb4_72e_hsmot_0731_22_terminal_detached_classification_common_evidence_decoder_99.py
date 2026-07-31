"""0731_22: gradient-isolated terminal classification common evidence.

This is the detached-gate counterpart of 0731_19. The terminal common
evidence still drives a trainable classification gate, but the added branch
cannot backpropagate into the two frame-evidence tensors. Box regression,
auxiliary predictions, and recurrent references remain on the 0727_01 path.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_pyramidlocal_pairdn_paircoherent_le180_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403

model['encoder']['post_pair_temporal_adapter_cfg'].update(
    type='pyramid_dual_evidence', use_spatial_evidence=False)
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
    terminal_classification_common_evidence_decoder=True,
    terminal_factorized_evidence_decoder=False,
    terminal_detach_gate_evidence=True)

work_dir = ('/data4/litianhao/PairMmot/workdir_99/'
            '0731_22_paper_base_liquid_encoder_p5temporal_dualevidence_'
            'decoder_terminaldetachedclassificationcommonevidence_'
            'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/wangying01/lth/PairMOT/TrackEval',
    track_data_root='/data/users/wangying01/lth/PairMOT/data/hsmot/test')
test_evaluator = val_evaluator
