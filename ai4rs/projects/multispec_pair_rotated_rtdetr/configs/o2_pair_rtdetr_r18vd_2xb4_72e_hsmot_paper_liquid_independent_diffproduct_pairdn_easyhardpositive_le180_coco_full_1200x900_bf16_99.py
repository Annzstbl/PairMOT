"""0728_02: corrected 0718_01 with easy/hard-positive PairDN.

The representation, geometry, and missing-side fixes of 0723_01 are retained.
Each logical DN group contains one easy-positive and one hard-positive query
for every GT, with independently sampled noise for the two frames. There are
no background-negative DN queries, and unused capacity remains masked.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_independent_diffproduct_pairdn_paircoherent_le180_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


model['pair_dn_cfg'].update(
    dn_target_mode='easy_hard_positive',
    share_pair_noise=False,
    positive_hard_min_magnitude=0.5,
    positive_hard_max_magnitude=1.25)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0728_02_paper_liquid_independent_diffproduct_pairdn_easyhardpositive_'
    'le180_r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
