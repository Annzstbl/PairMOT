"""0722_01: 0718_01 rerun with corrected DINO negative DN sampling."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_independent_diffproduct_accuracyfix_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0722_01_paper_liquid_independent_diffproduct_dnnegativefix_r18_coco_'
    'full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
