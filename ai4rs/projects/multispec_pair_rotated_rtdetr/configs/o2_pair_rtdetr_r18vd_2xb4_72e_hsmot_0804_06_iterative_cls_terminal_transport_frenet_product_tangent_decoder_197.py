"""0804_06 on 197: constant-turn Frenet product-tangent transport.

Relative to body-frame product tangent, only the center detail projector
changes. Reference orientation change rotates the chord into previous/current
endpoint tangents of a constant-turn arc, and each frame detail is projected
onto its own endpoint direction. Shape transport, classification, DN,
auxiliary outputs, recursive references, layers, attention, and losses are
unchanged. The operation is parameter-free, swap-equivariant, class agnostic,
and contains no reweighting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0804_03_iterative_cls_terminal_log_size_decoder_197 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_log_size_refinement_decoder=False,
    pair_shared_terminal_transport_center_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_shape_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_body_frame_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_se2_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_frenet_product_tangent_refinement_decoder=True,
    pair_shared_terminal_transport_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_plane_refinement_decoder=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0804_06_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransportfrenet_'
    'producttangentrefinement_pairdn_paircoherent_le180_r18_coco_full_'
    '1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
