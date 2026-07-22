"""0720_02: 0718_01 Liquid with response-mass fusion conservation."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_independent_diffproduct_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


model['backbone']['liquid_sampler']['liquid_aware_fusion'][
    'quality_conservation'] = dict(mode='response_mass')

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0720_02_paper_liquid_diffproduct_qc_responsemass_r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
