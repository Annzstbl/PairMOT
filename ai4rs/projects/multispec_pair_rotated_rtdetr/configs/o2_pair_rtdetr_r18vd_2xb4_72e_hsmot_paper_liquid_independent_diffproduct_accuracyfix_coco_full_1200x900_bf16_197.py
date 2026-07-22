"""0721_02: accuracy-fixed strict rerun of the 0718_01 Liquid model."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_independent_diffproduct_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


# Keep the 0718_01 fusion unrestricted. This run isolates the current paired
# geometry/GMC and PairDN correctness fixes from later conservation variants.
model['backbone']['liquid_sampler']['liquid_aware_fusion'].pop(
    'quality_conservation', None)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0721_02_paper_liquid_independent_diffproduct_accuracyfix_r18_coco_full_'
    '1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
