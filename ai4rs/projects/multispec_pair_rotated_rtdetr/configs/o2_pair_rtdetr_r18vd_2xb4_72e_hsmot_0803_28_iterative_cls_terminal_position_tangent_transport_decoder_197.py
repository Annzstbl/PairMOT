"""0803_28 on 197: position-tangent evidence plus full tangent geometry.

The final classification features retain only swap-odd cross-attention detail
aligned with the detached positional displacement.  Regression keeps the
coupled five-dimensional transported tangent that produced the strongest
early full-tangent result.  The operation is terminal-only, parameter-free,
swap-equivariant, class agnostic, and preserves the DN prefix.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_24_iterative_cls_terminal_transport_shape_tangent_decoder_197 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_transport_shape_tangent_refinement_decoder=False,
    terminal_position_tangent_transport_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0803_28_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_terminalpositiontangenttransport_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
