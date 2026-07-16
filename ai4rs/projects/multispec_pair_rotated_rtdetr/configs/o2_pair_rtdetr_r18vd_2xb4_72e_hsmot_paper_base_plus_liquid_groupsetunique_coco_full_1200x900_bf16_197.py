"""0716_04 Liquid run with globally unique hard group band sets on 197."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


model['backbone']['liquid_sampler']['hard_group_unique_sets'] = True

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0716_04_paper_base_plus_liquid_groupsetunique_r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
