"""0718_05 pair-consistent detail-preserving Liquid on 99."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_anchorcompetitive_adaptiveanchor_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


model['backbone']['liquid_sampler'][
    'pair_consistent_detail_preservation'] = dict(hidden_dims=16)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0718_05_paper_liquid_adaptiveanchor_pcdp_r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
