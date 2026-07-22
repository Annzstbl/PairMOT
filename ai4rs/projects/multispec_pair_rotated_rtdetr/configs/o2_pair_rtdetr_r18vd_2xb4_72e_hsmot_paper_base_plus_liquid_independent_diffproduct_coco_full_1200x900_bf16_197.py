"""0718_01 Liquid with independent groups and explicit pair relations."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


_sampler = model['backbone']['liquid_sampler']

# Each group chooses its three bands independently. Sampling within one group
# remains without replacement, so repeated forms such as 227 are still barred.
_sampler.update(
    hard_group_unique_sets=False,
    soft_group_set_transport=None,
    use_soft_context_after_hard=False)

# Expose signed difference and multiplicative agreement explicitly to both
# pair-coupling paths instead of asking an MLP to recover them from [x, y].
_sampler['pair_sampler_router']['relation_mode'] = 'pair_diff_product'
_sampler['liquid_aware_fusion']['pair_transport'][
    'relation_mode'] = 'pair_diff_product'

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0718_01_paper_base_plus_liquid_independent_diffproduct_r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
