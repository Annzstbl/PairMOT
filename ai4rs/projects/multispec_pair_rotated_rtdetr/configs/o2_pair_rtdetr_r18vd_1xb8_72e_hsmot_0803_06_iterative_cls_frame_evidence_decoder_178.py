"""0803_06: iterative cls reads existing frame-specific decoder evidence.

The shared recurrent query, iterative references, regression, and DN remain on
the 0801_09 parent path. At every decoder layer, only the iterative
classification residual heads receive the already-computed prev/curr
cross-attention evidence instead of the post-fusion shared state. This restores
visibility-specific evidence without an adapter, new parameter, attention,
loss, class-aware route, or score reweighting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0801_09_iterative_cls_dn_isolated_e2e_decoder_178 import *  # noqa: F401,F403


model['decoder'].update(
    frame_evidence_cls_decoder=True,
    pair_shared_shape_refinement_decoder=False,
    pair_shared_angle_refinement_decoder=False,
    pair_shared_periodic_angle_refinement_decoder=False,
    pair_shared_normalized_center_refinement_decoder=False)
model['bbox_head'].update(
    iterative_cls_pair_shared_objectness=False,
    iterative_cls_residual=True,
    iterative_cls_dn_absolute=True,
    iterative_cls_detach_between_layers=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0803_06_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_frameevidencecls_pairdn_'
    'paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
