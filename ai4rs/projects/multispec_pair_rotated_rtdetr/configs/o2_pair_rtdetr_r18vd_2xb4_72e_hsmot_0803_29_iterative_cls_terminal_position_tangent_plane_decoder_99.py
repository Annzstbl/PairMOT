"""0803_29 on 99: position evidence plus tangent-plane geometry.

Terminal classification keeps only position-aligned swap-odd evidence.  Box
detail is projected onto the local plane spanned by established pair motion
and the detached pair-common terminal correction.  This relaxes the mature
one-dimensional transport bottleneck without adding parameters, states,
attention, losses, class-aware rules, or reweighting.  Auxiliary layers and
the DN prefix remain on the parent path.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_27_iterative_cls_terminal_position_tangent_product_decoder_99 import *  # noqa: F401,F403


model['decoder'].update(
    terminal_position_tangent_product_decoder=False,
    terminal_position_tangent_plane_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0803_29_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_terminalpositiontangentplane_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
