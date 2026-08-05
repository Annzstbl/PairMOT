"""0804_13 on 178: hemisphere-fold center with log-shape consensus.

This is the single-GPU, physical-batch-8 equivalent of the 99 two-GPU
candidate. It reflects only center detail that points against established
translation while retaining every transverse component and the complete
detail norm. Classification, DN, losses, attention, layers, recurrent
references, and the mature terminal log-shape consensus are unchanged.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0804_12_iterative_cls_terminal_transport_spherical_midpoint_center_log_shape_consensus_decoder_178 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_transport_spherical_midpoint_center_log_shape_consensus_refinement_decoder=False,
    pair_shared_terminal_transport_hemisphere_fold_center_log_shape_consensus_refinement_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0804_13_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransport_'
    'hemispherefoldcenter_logshapeconsensus_pairdn_paircoherent_le180_'
    'r18_coco_full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
