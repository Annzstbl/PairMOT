"""0717_03 hard-sampled, soft-context Liquid on the paper protocol."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_groupsetunique_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


# Keep globally unique hard bands for the Conv3D path while preserving the
# corresponding continuous probabilities for GroupMod and Liquid-aware fusion.
model['backbone']['liquid_sampler']['use_soft_context_after_hard'] = True

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0717_03_paper_base_plus_liquid_hardsoftcontext_r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
