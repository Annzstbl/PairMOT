"""0718_02 collapse-resistant, content-adaptive paper Liquid candidate."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


_sampler = model['backbone']['liquid_sampler']
_sampler.update(
    hard_group_unique_sets=False,
    soft_group_set_transport=None,
    use_soft_context_after_hard=False,
    competitive_router=dict(
        content_dims=24,
        content_strength=0.35,
        common_cap=0.5,
        specific_cap=2.0))
_sampler['pair_sampler_router']['relation_mode'] = 'pair_diff_product'
_sampler['liquid_aware_fusion']['pair_transport'][
    'relation_mode'] = 'pair_diff_product'

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0718_02_paper_base_plus_liquid_anchorcompetitive_r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
