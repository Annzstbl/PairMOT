"""0731_10: midpoint-preserving regression-only frame detail.

Frame evidence is converted into one antisymmetric correction in 5D
box-logit residual space. The added correction therefore has an exactly zero
pair midpoint even with independent nonlinear regression heads, while the
classification path remains shared.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_decoder0708_03_pairdn_paircoherent_le180_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


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
    regression_enveloped_detail_decoder=False,
    midpoint_regression_enveloped_detail_decoder=True,
    classification_enveloped_detail_decoder=False,
    common_evidence_bypass_decoder=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0731_10_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_sharedattention_midpointregressionenvelopeddetail_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao/PairMOT_sync_3cb888d/TrackEval',
    track_data_root='/data/users/litianhao/PairMOT/data/hsmot/test')
test_evaluator = val_evaluator
