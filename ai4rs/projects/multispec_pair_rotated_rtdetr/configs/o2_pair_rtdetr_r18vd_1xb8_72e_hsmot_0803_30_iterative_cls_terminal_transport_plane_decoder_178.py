"""0803_30: geometry-only terminal osculating-plane transport.

Relative to 0803_23, only the final normal-query box-detail projection changes
from the established one-dimensional motion tangent to the at-most
two-dimensional plane spanned by established motion and the detached
pair-common terminal correction.  Classification, DN, auxiliary outputs, and
recursive references retain the parent path.  The projection adds no
parameters, states, layers, attention, losses, class-aware rules, or
reweighting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0803_23_iterative_cls_terminal_transport_tangent_decoder_178 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_transport_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_plane_refinement_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0803_30_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransportplane_'
    'refinement_pairdn_paircoherent_le180_r18_coco_full_1200x900_'
    'bf16_1xb8_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
