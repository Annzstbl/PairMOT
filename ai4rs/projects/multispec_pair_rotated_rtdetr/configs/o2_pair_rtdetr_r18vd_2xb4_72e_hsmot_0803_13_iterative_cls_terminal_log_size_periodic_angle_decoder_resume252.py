"""252-portable continuation of mature 0803_13 from epoch 24.

The scientific model and global batch remain identical to the active 178
1x8 trajectory. Only physical placement changes to fixed 252 GPU0/1 with
2x4, allowing the mature branch to continue while 178 explores a successor.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_14_iterative_cls_terminal_log_area_periodic_angle_decoder_252 import *  # noqa: F401,F403


model['decoder'].update(
    frame_evidence_cls_decoder=False,
    frame_detail_cls_decoder=False,
    pair_shared_shape_refinement_decoder=False,
    pair_shared_angle_refinement_decoder=False,
    pair_shared_periodic_angle_refinement_decoder=False,
    pair_shared_log_size_periodic_angle_refinement_decoder=False,
    pair_shared_log_area_periodic_angle_refinement_decoder=False,
    pair_shared_late_log_size_periodic_angle_refinement_decoder=False,
    pair_shared_terminal_log_size_periodic_angle_refinement_decoder=True,
    pair_shared_terminal_log_area_periodic_angle_refinement_decoder=False,
    pair_shared_terminal_periodic_angle_refinement_decoder=False,
    pair_shared_terminal_normalized_center_refinement_decoder=False,
    pair_shared_terminal_full_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_tangent_refinement_decoder=False,
    pair_shared_progressive_log_shape_periodic_angle_refinement_decoder=False,
    pair_shared_normalized_center_refinement_decoder=False)
model['bbox_head'].update(
    iterative_cls_pair_shared_objectness=False,
    iterative_cls_terminal_shared_margins=False,
    iterative_cls_terminal_transport_margins=False,
    iterative_cls_residual=True,
    iterative_cls_dn_absolute=True,
    iterative_cls_detach_between_layers=False)

# The 252 parent explicitly carries several disabled switches that are absent
# from the authoritative 178 config. Remove them so the effective scientific
# model dict is byte-for-byte equivalent before resuming the optimizer state.
model['decoder'].pop('residual_preserving_fusion_decoder', None)
model['decoder'].pop(
    'pair_shared_terminal_log_area_periodic_angle_refinement_decoder', None)
model['decoder'].pop(
    'pair_shared_terminal_periodic_angle_refinement_decoder', None)
model['decoder'].pop(
    'pair_shared_terminal_normalized_center_refinement_decoder', None)
model['decoder'].pop(
    'pair_shared_terminal_full_tangent_refinement_decoder', None)
model['decoder'].pop(
    'pair_shared_terminal_transport_tangent_refinement_decoder', None)
model['bbox_head'].pop('iterative_cls_terminal_shared_margins', None)
model['bbox_head'].pop('iterative_cls_terminal_transport_margins', None)
model['bbox_head'].pop('terminal_encoder_cls_residual', None)
model['bbox_head'].pop('terminal_pair_common_cls_residual', None)
model['bbox_head'].pop('terminal_pair_common_objectness_residual', None)
model['bbox_head'].pop('terminal_pair_differential_objectness_residual', None)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0803_13_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminallogsizetangent_'
    'periodicanglerefinement_pairdn_paircoherent_le180_r18_coco_full_'
    '1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao01/PairMmot/TrackEval',
    track_data_root='/data/users/litianhao01/PairMmot/data/hsmot/test')
test_evaluator = val_evaluator
