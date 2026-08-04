"""0804_05 on 99: terminal SE(2) midpoint Lie-twist transport.

The transported shape tangent is unchanged from 0804_01. Only the terminal
center correction is lifted from displacement space to a finite-motion SE(2)
midpoint tangent using the already transported angle increment, projected
along the reference trajectory, and retracted with the matching Jacobian.
Classification, DN, auxiliary outputs, recursive references, layers,
attention, and losses are unchanged. The operation is parameter-free,
swap-equivariant, class agnostic, and contains no reweighting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0804_02_iterative_cls_terminal_periodic_angle_decoder_99 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_periodic_angle_refinement_decoder=False,
    pair_shared_terminal_transport_center_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_shape_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_body_frame_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_se2_product_tangent_refinement_decoder=True,
    pair_shared_terminal_transport_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_plane_refinement_decoder=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0804_05_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransportse2_'
    'producttangentrefinement_pairdn_paircoherent_le180_r18_coco_full_'
    '1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
