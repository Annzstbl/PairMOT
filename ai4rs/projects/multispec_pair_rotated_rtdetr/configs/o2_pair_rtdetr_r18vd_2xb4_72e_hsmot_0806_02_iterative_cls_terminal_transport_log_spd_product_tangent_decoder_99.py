"""0806_02 on 99: log-Euclidean SPD shape product-tangent.

This two-GPU port keeps the 0804_01 center tangent and changes only the shape
metric to axis-relabeling-invariant log-SPD coordinates. It is parameter free,
class agnostic, swap equivariant, and has constant terminal elementwise cost.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_26_iterative_cls_terminal_transport_product_tangent_decoder_99 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_transport_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_log_spd_product_tangent_refinement_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0806_02_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransport_'
    'logspd_producttangent_pairdn_paircoherent_le180_r18_coco_'
    'full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
