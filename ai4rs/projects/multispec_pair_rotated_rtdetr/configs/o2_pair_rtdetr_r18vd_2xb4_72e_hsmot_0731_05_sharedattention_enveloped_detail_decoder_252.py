"""0731_05: shared attention plus evidence-enveloped frame detail.

This composes the shared geometric aggregation policy that improved early
association with the bounded head-local frame correction that preserved
detection better than the unconstrained antisymmetric branch.  Sampling
offsets and value/output projections remain frame-specific, the recurrent
query path is unchanged, and the zero-start detail magnitude cannot exceed
the raw cross-attention frame difference.
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
    shared_attention_decoder=True,
    antisymmetric_detail_decoder=False,
    enveloped_detail_decoder=True,
    common_evidence_bypass_decoder=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0731_05_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_sharedattention_envelopeddetail_pairdn_paircoherent_le180_r18_'
    'coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao01/PairMmot/TrackEval',
    track_data_root='/data/users/litianhao01/PairMmot/data/hsmot/test')
test_evaluator = val_evaluator
