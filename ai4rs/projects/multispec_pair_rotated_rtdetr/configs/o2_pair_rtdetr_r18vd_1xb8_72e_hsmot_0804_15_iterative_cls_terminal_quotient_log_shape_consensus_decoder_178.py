"""0804_15 on 178: quotient-aware terminal log-shape consensus.

The terminal width/height tangent is aligned across the rotated-box
equivalence (w, h, theta) ~ (h, w, theta + pi/2) before pair averaging.
Centers, classification, DN, losses, attention, layers, and recurrent
references remain unchanged. The operation is parameter free, class
agnostic, frame-swap equivariant, and adds only constant elementwise work.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0804_14_iterative_cls_terminal_transport_hemisphere_boundary_center_log_shape_consensus_decoder_178 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_transport_hemisphere_boundary_center_log_shape_consensus_refinement_decoder=False,
    pair_shared_terminal_quotient_log_size_periodic_angle_refinement_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0804_15_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminalquotient_'
    'logshapeconsensus_pairdn_paircoherent_le180_r18_coco_full_'
    '1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
