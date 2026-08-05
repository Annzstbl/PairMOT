"""0804_11 on 99: center tangent with mature log-shape consensus.

Relative to the mature terminal log-size/periodic-angle consensus, the only
new factor is a terminal product-tangent projection of antisymmetric center
detail.  Shape detail remains a complete terminal consensus.  Classification,
DN, losses, attention, layers, and recurrent references are unchanged.  The
operation is parameter-free, swap-equivariant, class agnostic, and has no
reweighting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0804_08_iterative_cls_terminal_transport_shared_metric_product_tangent_decoder_99 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_log_size_periodic_angle_refinement_decoder=False,
    pair_shared_terminal_transport_center_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_center_tangent_log_shape_consensus_refinement_decoder=True,
    pair_shared_terminal_transport_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_shared_metric_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_householder_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_body_frame_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_se2_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_frenet_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_axis_frenet_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_covariant_frenet_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_shape_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_plane_refinement_decoder=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0804_11_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransportcenter_'
    'tangent_logshapeconsensus_pairdn_paircoherent_le180_r18_coco_full_'
    '1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
