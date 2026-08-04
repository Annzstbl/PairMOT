"""0803_25 on 99: terminal transported center-tangent consensus."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_21_iterative_cls_terminal_transport_margins_decoder_99 import *  # noqa: F401,F403


model['decoder'].update(
    frame_evidence_cls_decoder=False,
    frame_detail_cls_decoder=False,
    pair_shared_shape_refinement_decoder=False,
    pair_shared_angle_refinement_decoder=False,
    pair_shared_periodic_angle_refinement_decoder=False,
    pair_shared_log_size_periodic_angle_refinement_decoder=False,
    pair_shared_log_area_periodic_angle_refinement_decoder=False,
    pair_shared_late_log_size_periodic_angle_refinement_decoder=False,
    pair_shared_terminal_log_size_periodic_angle_refinement_decoder=False,
    pair_shared_terminal_log_area_periodic_angle_refinement_decoder=False,
    pair_shared_terminal_periodic_angle_refinement_decoder=False,
    pair_shared_terminal_normalized_center_refinement_decoder=False,
    pair_shared_terminal_full_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_center_tangent_refinement_decoder=True,
    pair_shared_terminal_transport_shape_tangent_refinement_decoder=False,
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

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0803_25_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransportcenter_'
    'tangentrefinement_pairdn_paircoherent_le180_r18_coco_full_1200x900_'
    'bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
