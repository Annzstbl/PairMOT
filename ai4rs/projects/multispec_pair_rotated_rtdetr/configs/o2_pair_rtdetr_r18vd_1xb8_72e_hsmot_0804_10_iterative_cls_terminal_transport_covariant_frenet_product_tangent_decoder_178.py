"""0804_10 on 178: covariant Frenet product-tangent transport.

Relative to 0804_01, only terminal center transport changes. Previous and
current center updates are parallel-transported into the midpoint Frenet
frame before common/detail decomposition. The complete common two-vector is
retained, while only transverse aligned inconsistency is removed. Shape
transport, classification, DN, auxiliary outputs, recursive references,
layers, attention, and losses are unchanged. The operation is parameter-free,
swap-equivariant, class agnostic, and contains no gate or reweighting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0804_01_iterative_cls_terminal_transport_product_tangent_decoder_178 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_transport_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_covariant_frenet_product_tangent_refinement_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0804_10_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransportcovariant_'
    'frenet_producttangentrefinement_pairdn_paircoherent_le180_r18_coco_'
    'full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
