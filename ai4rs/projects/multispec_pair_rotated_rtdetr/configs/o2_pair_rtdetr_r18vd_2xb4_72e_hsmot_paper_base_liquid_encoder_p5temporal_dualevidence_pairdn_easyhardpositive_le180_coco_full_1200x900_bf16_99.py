"""0728_03: 0728_02 PairDN plus the 0727_01 Dual-Evidence encoder.

Relative to 0728_02, this adds only the validated P5 temporal MHA and the
post-FPN dual common/detail evidence adapter used by 0727_01.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_pyramidlocal_pairdn_paircoherent_le180_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


model['pair_dn_cfg'].update(
    dn_target_mode='easy_hard_positive',
    share_pair_noise=False,
    positive_hard_min_magnitude=0.5,
    positive_hard_max_magnitude=1.25)
model['encoder']['post_pair_temporal_adapter_cfg'].update(
    type='pyramid_dual_evidence',
    use_spatial_evidence=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0728_03_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'pairdn_easyhardpositive_le180_r18_coco_full_1200x900_bf16_'
    'orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
