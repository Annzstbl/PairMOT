"""0730_13: shared attention weights with frame-specific localization.

The two adjacent-frame deformable cross-attention branches share only their
attention-weight predictor.  Sampling offsets, value projections, and output
projections remain independent.  This retains the association prior tested by
0730_11 without removing the frame-specific localization freedom whose loss
reduced pair mAP.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_dualoutputresidual_pairdn_paircoherent_le180_coco_full_1200x900_bf16_252 import *  # noqa: F401,F403


model['decoder'].update(
    tristate_decoder=False,
    dual_output_adapter=False,
    common_motion_decoder=False,
    shared_evidence_decoder=False,
    competitive_evidence_decoder=False,
    motion_trust_decoder=False,
    symmetric_pair_decoder=False,
    shared_routing_decoder=False,
    shared_attention_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0730_13_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_sharedattention_pairdn_paircoherent_le180_r18_'
    'coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao01/PairMmot/TrackEval',
    track_data_root='/data/users/litianhao01/PairMmot/data/hsmot/test')
test_evaluator = val_evaluator
