"""0803_01: iterative pair-shared objectness residual refinement.

Each frame retains an independent class-margin residual at every decoder
layer.  Only the mean residual over classes is replaced by the two-frame
mean, sharing a class-permutation-equivariant objectness correction without
parameters, learned weights, class identities, extra attention, or losses.
DN queries remain on the original absolute classifiers.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0801_07_iterative_cls_residual_decoder_252 import *  # noqa: F401,F403


model['decoder'].update(
    symmetric_feature_decoder=False,
    symmetric_pair_decoder=False,
    symmetric_position_decoder=False,
    residual_preserving_fusion_decoder=False)
model['bbox_head'].update(
    iterative_cls_residual=True,
    iterative_cls_dn_absolute=True,
    iterative_cls_detach_between_layers=False,
    iterative_cls_pair_shared_objectness=True,
    terminal_encoder_cls_residual=False,
    terminal_pair_common_cls_residual=False,
    terminal_pair_common_objectness_residual=False,
    terminal_pair_differential_objectness_residual=False,
    cls_proto_gate=False,
    cls_residual_adapter=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0803_01_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclspairsharedobjectnessdnisolatede2e_pairdn_'
    'paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao01/PairMmot/TrackEval',
    track_data_root='/data/users/litianhao01/PairMmot/data/hsmot/test')
test_evaluator = val_evaluator
