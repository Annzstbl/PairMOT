"""0717_02 paper Base + Liquid with the original per-group hard sampler."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


# Preserve the Liquid variant that improved the historical full-data baseline:
# hard sampling removes duplicate bands only within each group, while groups
# remain free to select the same unordered band set.
model['backbone']['liquid_sampler'].update(
    hard_group_unique_sets=False,
    soft_group_set_transport=None)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0717_02_paper_base_plus_liquid_originalhard_r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
