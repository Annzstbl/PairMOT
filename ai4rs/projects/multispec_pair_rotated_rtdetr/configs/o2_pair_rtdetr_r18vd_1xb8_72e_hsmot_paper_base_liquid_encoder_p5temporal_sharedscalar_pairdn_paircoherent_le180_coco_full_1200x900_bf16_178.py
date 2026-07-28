"""0727_06: pair-shared scalar evidence gain plus signed detail.

Common evidence may only apply a positive, bounded scalar gain shared by both
frames at each spatial location. It cannot rotate or mix feature channels,
while the existing odd detail branch remains responsible for pair-specific
temporal correction. Liquid and the complete paper protocol are unchanged.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178 import *  # noqa: F401,F403


model['encoder']['post_pair_temporal_adapter_cfg'].update(
    type='pyramid_common_detail',
    use_shared_scalar_gain=True)
model['encoder']['post_pair_temporal_adapter_cfg'].pop(
    'use_spatial_evidence', None)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0727_06_paper_base_liquid_encoder_p5temporal_sharedscalar_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
