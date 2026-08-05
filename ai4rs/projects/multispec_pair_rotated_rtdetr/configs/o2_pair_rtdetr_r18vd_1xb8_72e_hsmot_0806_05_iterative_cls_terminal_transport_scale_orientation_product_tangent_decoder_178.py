"""0806_05 on 178: scale-orientation split product-tangent.

The successful 0804_01 center tangent is unchanged. Rectangle shape is
treated as the product of a two-dimensional log-size plane and the periodic
orientation circle. Only scale detail is projected along the reference scale
transport; each frame keeps its proposed orientation. This removes the raw
shape tangent's log-scale/angle unit competition without coupling area and
anisotropy as in log-SPD or quotient-anisotropy alternatives. The operation
is parameter free, class agnostic, swap equivariant, and adds only constant
terminal elementwise work.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0804_01_iterative_cls_terminal_transport_product_tangent_decoder_178 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_transport_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_scale_orientation_product_tangent_refinement_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0806_05_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransport_'
    'scaleorientation_producttangent_pairdn_paircoherent_le180_r18_coco_'
    'full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
