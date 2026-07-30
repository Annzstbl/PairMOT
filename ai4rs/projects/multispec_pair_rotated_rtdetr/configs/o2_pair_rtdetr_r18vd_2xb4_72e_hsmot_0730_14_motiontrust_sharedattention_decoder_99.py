"""0730_14: motion-trust plus frame-localized shared attention.

Motion-trust protects detection geometry with a confidence-gated bounded
antisymmetric correction.  Shared attention removes frame-specific weighting
bias while retaining independent sampling offsets and value/output
projections.  The combination tests whether these orthogonal geometry and
aggregation priors reinforce one another.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_pyramidlocal_pairdn_paircoherent_le180_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


model['encoder']['post_pair_temporal_adapter_cfg'].update(
    type='pyramid_dual_evidence',
    use_spatial_evidence=False)
model['decoder'].update(
    tristate_decoder=False,
    tristate_separate_ffn=False,
    tristate_zero_init_coupling=False,
    dual_output_adapter=False,
    common_motion_decoder=False,
    shared_evidence_decoder=False,
    competitive_evidence_decoder=False,
    motion_trust_decoder=True,
    symmetric_pair_decoder=False,
    shared_routing_decoder=False,
    shared_attention_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0730_14_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_motiontrust_sharedattention_pairdn_paircoherent_le180_r18_'
    'coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/wangying01/lth/PairMOT/TrackEval',
    track_data_root='/data/users/wangying01/lth/PairMOT/data/hsmot/test')
test_evaluator = val_evaluator
