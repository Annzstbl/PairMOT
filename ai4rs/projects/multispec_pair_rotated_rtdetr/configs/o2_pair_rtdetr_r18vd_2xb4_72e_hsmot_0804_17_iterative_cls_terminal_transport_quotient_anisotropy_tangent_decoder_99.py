"""0804_17 on 99: quotient-anisotropy product-tangent transport.

This is the two-GPU port of the 178 candidate. It keeps the successful
factorized center tangent and replaces only the mixed log-width/log-height/
angle shape projection with axis-relabeling-invariant double-angle
anisotropy transport. Each frame's proposed log area remains unchanged.
Classification, DN, losses, attention, layers, and recurrent references are
unchanged. The operation is parameter free, class agnostic, swap equivariant,
and adds only constant terminal elementwise work.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_26_iterative_cls_terminal_transport_product_tangent_decoder_99 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_transport_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_quotient_anisotropy_tangent_refinement_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0804_17_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransport_'
    'quotientanisotropytangent_pairdn_paircoherent_le180_r18_coco_'
    'full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
