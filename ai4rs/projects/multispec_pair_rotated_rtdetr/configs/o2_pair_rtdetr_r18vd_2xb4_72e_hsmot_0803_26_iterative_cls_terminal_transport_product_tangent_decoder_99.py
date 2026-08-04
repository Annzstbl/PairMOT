"""0803_26 on 99: factorized product-tangent terminal transport.

Translation and shape updates are projected in independent 2D and 3D tangent
bundles. This keeps the complete transported geometry of 0803_23 while
preventing center magnitude from rotating into log-size/angle detail through a
single 5D projection. The operation is terminal-only, parameter-free,
swap-equivariant, class agnostic, and leaves DN queries unchanged.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_25_iterative_cls_terminal_transport_center_tangent_decoder_99 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_transport_center_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_shape_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_product_tangent_refinement_decoder=True,
    pair_shared_terminal_transport_tangent_refinement_decoder=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0803_26_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransportproduct_'
    'tangentrefinement_pairdn_paircoherent_le180_r18_coco_full_1200x900_'
    'bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
