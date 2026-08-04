"""0804_01 on 178: factorized product-tangent terminal transport.

Relative to the mature 0803_23 full tangent, terminal pair detail is projected
independently in the 2D center and 3D log-size/periodic-angle tangent bundles.
This removes cross-block inner-product leakage while retaining the full
pair-common correction. Classification, DN, auxiliary outputs, and recursive
references are unchanged. The operation is parameter-free, swap-equivariant,
class agnostic, and adds no layer, attention, loss, or reweighting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0803_25_iterative_cls_terminal_transport_center_tangent_decoder_178 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_transport_center_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_shape_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_product_tangent_refinement_decoder=True,
    pair_shared_terminal_transport_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_plane_refinement_decoder=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0804_01_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransportproduct_'
    'tangentrefinement_pairdn_paircoherent_le180_r18_coco_full_1200x900_'
    'bf16_1xb8_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
