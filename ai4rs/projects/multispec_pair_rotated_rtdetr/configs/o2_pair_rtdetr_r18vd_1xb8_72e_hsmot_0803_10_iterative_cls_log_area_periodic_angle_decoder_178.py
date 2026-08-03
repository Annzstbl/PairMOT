"""0803_10: shared log-area plus periodic-angle decoder refinement.

Normal-query width/height refinement shares only the isotropic log-area
increment.  Each frame keeps its own log-aspect increment and applies both
components relative to its own reference.  Orientation uses the pi-periodic
tangent midpoint.  Centers, classification, recurrent queries, DN, losses,
attention count, and decoder depth remain on the 0801_09 parent path.  The
projection is parameter-free and class-agnostic.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0801_09_iterative_cls_dn_isolated_e2e_decoder_178 import *  # noqa: F401,F403


model['decoder'].update(
    frame_evidence_cls_decoder=False,
    frame_detail_cls_decoder=False,
    pair_shared_shape_refinement_decoder=False,
    pair_shared_angle_refinement_decoder=False,
    pair_shared_periodic_angle_refinement_decoder=False,
    pair_shared_log_size_periodic_angle_refinement_decoder=False,
    pair_shared_log_area_periodic_angle_refinement_decoder=True,
    pair_shared_normalized_center_refinement_decoder=False)
model['bbox_head'].update(
    iterative_cls_pair_shared_objectness=False,
    iterative_cls_residual=True,
    iterative_cls_dn_absolute=True,
    iterative_cls_detach_between_layers=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0803_10_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedlogarea_'
    'periodicanglerefinement_pairdn_paircoherent_le180_'
    'r18_coco_full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
