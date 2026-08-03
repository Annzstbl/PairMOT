"""0803_07 DDP: frame-local classification plus periodic angle consensus.

Classification reads each frame's existing cross-attention evidence, while
normal-query angle refinement uses the pi-periodic tangent-space midpoint.
The two zero-parameter routes are orthogonal: shared recurrent queries, all
other box coordinates, DN, losses, attention count, and decoder depth stay on
the 0801_09 parent path.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_03_iterative_cls_pair_shared_angle_refinement_decoder_252 import *  # noqa: F401,F403


model['decoder'].update(
    frame_evidence_cls_decoder=True,
    pair_shared_shape_refinement_decoder=False,
    pair_shared_angle_refinement_decoder=False,
    pair_shared_periodic_angle_refinement_decoder=True,
    pair_shared_normalized_center_refinement_decoder=False)
model['bbox_head'].update(
    iterative_cls_pair_shared_objectness=False,
    iterative_cls_residual=True,
    iterative_cls_dn_absolute=True,
    iterative_cls_detach_between_layers=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0803_07_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_frameevidencecls_'
    'pairsharedperiodicanglerefinement_pairdn_paircoherent_le180_'
    'r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao01/PairMmot/TrackEval',
    track_data_root='/data/users/litianhao01/PairMmot/data/hsmot/test')
test_evaluator = val_evaluator
