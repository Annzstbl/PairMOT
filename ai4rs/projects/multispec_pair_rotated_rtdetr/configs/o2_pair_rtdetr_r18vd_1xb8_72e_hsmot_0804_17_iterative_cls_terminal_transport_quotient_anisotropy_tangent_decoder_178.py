"""0804_17 on 178: quotient-anisotropy product-tangent transport.

This keeps the successful 0804_01 center tangent but replaces its mixed
log-width/log-height/angle shape projection with a physical rectangle
quotient. Pair detail is transported only in the double-angle anisotropy
plane, while each frame's proposed log area remains unchanged. The single
structural change removes axis-label ambiguity and avoids mixing scale with
orientation. Classification, DN, losses, attention, layers, and recurrent
references remain unchanged. The operation is parameter free, class
agnostic, swap equivariant, and adds only constant terminal elementwise work.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0804_01_iterative_cls_terminal_transport_product_tangent_decoder_178 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_transport_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_quotient_anisotropy_tangent_refinement_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0804_17_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransport_'
    'quotientanisotropytangent_pairdn_paircoherent_le180_r18_coco_'
    'full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
