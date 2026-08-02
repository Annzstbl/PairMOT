"""0803_03 single-GPU: iterative cls plus pair-shared angle refinement.

Normal queries retain independent x/y/w/h residuals while only their angle
residual uses the two-frame mean at every decoder layer. DN queries remain
untouched. The operation is zero-parameter, swap-equivariant, class agnostic,
and adds no reweighting, loss, attention, or decoder layer.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0801_09_iterative_cls_dn_isolated_e2e_decoder_178 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_shape_refinement_decoder=False,
    pair_shared_angle_refinement_decoder=True)
model['bbox_head'].update(iterative_cls_pair_shared_objectness=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0803_03_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedanglerefinement_pairdn_'
    'paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
