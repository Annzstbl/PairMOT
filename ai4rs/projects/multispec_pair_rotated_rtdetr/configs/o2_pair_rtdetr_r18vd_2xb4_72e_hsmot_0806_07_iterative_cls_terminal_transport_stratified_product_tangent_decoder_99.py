"""0806_07 on 99: zero-motion-safe stratified product tangent.

The direct product-tangent parent is unchanged on every non-degenerate
translation and shape-transport stratum. When an established reference
transport has numerically zero energy, its tangent axis is undefined, so this
variant preserves the proposed frame detail instead of collapsing it through
the projection epsilon. The terminal operation is parameter free, class
agnostic, swap equivariant, and leaves DN, classification, losses, attention,
layers, and recurrent references unchanged.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_26_iterative_cls_terminal_transport_product_tangent_decoder_99 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_transport_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_stratified_product_tangent_refinement_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0806_07_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransport_'
    'stratifiedproducttangent_pairdn_paircoherent_le180_r18_coco_'
    'full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
