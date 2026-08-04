"""0804_04 on 178: oriented body-frame product-tangent transport.

Relative to 0804_01, only the terminal center tangent bundle changes. Center
corrections and established translation are expressed in a shared object-local
frame defined by the pair's pi-periodic midpoint orientation and geometric
mean size before projection. The shape tangent, classification, DN, auxiliary
outputs, and recursive references are unchanged. This is parameter-free,
swap-equivariant, class agnostic, and adds no layer, attention, loss, or
reweighting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0804_01_iterative_cls_terminal_transport_product_tangent_decoder_178 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_transport_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_body_frame_product_tangent_refinement_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0804_04_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransportbodyframe_'
    'producttangentrefinement_pairdn_paircoherent_le180_r18_coco_full_'
    '1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
