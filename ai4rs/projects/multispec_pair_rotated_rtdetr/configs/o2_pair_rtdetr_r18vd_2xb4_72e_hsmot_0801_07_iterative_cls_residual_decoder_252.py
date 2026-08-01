"""0801_07: lightweight iterative decoder classification refinement.

The 0727_01 encoder, Liquid feature path, PairDN construction, data protocol,
decoder depth, losses, and regression path remain unchanged. Each of the
existing three decoder layers predicts a zero-initialized residual on the
detached classification logits from the preceding stage, mirroring iterative
box refinement. Six 256-to-8 linear projections add 12,336 parameters
(approximately 0.054%) and no attention, decoder layer, auxiliary loss, class
weighting, or tunable scale.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0801_06_symmetric_position_residual_preserving_fusion_decoder_252 import *  # noqa: F401,F403


model['decoder'].update(
    symmetric_position_decoder=False,
    symmetric_feature_decoder=False,
    residual_preserving_fusion_decoder=False)
model['bbox_head'].update(
    iterative_cls_residual=True,
    cls_proto_gate=False,
    cls_residual_adapter=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0801_07_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsresidual_pairdn_paircoherent_le180_r18_'
    'coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao01/PairMmot/TrackEval',
    track_data_root='/data/users/litianhao01/PairMmot/data/hsmot/test')
test_evaluator = val_evaluator
