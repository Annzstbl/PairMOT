"""0804_03 on 197: terminal-only multiplicative size consensus.

Only the final normal-query width/height increments share their log-domain
pair mean. Centers, periodic angle, classification, DN, auxiliary outputs,
and recurrent references remain frame-specific. Together with 0804_02
angle-only, this isolates which quotient coordinate supplies the mature
terminal-geometry gain without coupling scale and orientation. The operation
is parameter-free, swap-equivariant, class agnostic, and adds no layer,
attention, loss, or reweighting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_24_iterative_cls_terminal_transport_shape_tangent_decoder_197 import *  # noqa: F401,F403


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
    pair_shared_terminal_log_size_refinement_decoder=True,
    pair_shared_terminal_normalized_center_refinement_decoder=False,
    pair_shared_terminal_full_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_center_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_shape_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_plane_refinement_decoder=False,
    terminal_position_tangent_product_decoder=False,
    terminal_position_tangent_transport_decoder=False,
    terminal_position_tangent_plane_decoder=False,
    pair_shared_progressive_log_shape_periodic_angle_refinement_decoder=False,
    pair_shared_normalized_center_refinement_decoder=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0804_03_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminallogsize_'
    'refinement_pairdn_paircoherent_le180_r18_coco_full_1200x900_'
    'bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
