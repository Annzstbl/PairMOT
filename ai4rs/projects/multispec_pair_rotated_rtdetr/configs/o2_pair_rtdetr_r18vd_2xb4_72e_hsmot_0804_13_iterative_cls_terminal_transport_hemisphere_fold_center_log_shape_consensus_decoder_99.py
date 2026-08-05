"""0804_13 on 99: minimal hemisphere fold with log-shape consensus.

The mature terminal log-size/periodic-angle consensus is retained. The only
new factor reflects a center-detail component when and only when it points
against established pair translation. Already consistent and transverse
detail is unchanged, and the complete center-detail norm is preserved. The
operation is parameter-free, swap-equivariant, class agnostic, and has no
reweighting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0804_12_iterative_cls_terminal_transport_spherical_midpoint_center_log_shape_consensus_decoder_99 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_transport_spherical_midpoint_center_log_shape_consensus_refinement_decoder=False,
    pair_shared_terminal_transport_hemisphere_fold_center_log_shape_consensus_refinement_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0804_13_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransport_'
    'hemispherefoldcenter_logshapeconsensus_pairdn_paircoherent_le180_'
    'r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
