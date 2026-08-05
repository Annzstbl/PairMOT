"""0804_12 on 99: norm-preserving half-transport with log-shape consensus.

Relative to the mature terminal log-size/periodic-angle consensus, the only
new factor moves antisymmetric center-detail direction to the spherical
midpoint between its learned direction and established pair translation.
Magnitude, classification, DN, losses, attention, layers, and recurrent
references are unchanged. The operation is parameter-free, swap-equivariant,
class agnostic, and has no reweighting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0804_11_iterative_cls_terminal_transport_center_tangent_log_shape_consensus_decoder_99 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_transport_center_tangent_log_shape_consensus_refinement_decoder=False,
    pair_shared_terminal_transport_spherical_midpoint_center_log_shape_consensus_refinement_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0804_12_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransport_'
    'sphericalmidpointcenter_logshapeconsensus_pairdn_paircoherent_le180_'
    'r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
