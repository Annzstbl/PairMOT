"""0718_06 task-preserving adaptive Set-Transport Liquid candidate."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_settransport_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


# Preserve the detector-trained sampler logits and the strongest completed
# Set-Transport path. Pair-conditioned evidence may reorder only ambiguous
# candidates; confident task decisions receive a small bounded residual.
model['backbone']['liquid_sampler']['confidence_preserving_router'] = dict(
    content_dims=24,
    content_strength=0.2,
    margin_threshold=0.35,
    margin_temperature=0.1,
    min_gate=0.05,
    min_task_scale=0.25,
    max_task_scale=2.0)

# The strongest completed classification result used explicit signed change
# and agreement in both pair paths. Combine that relation with Set-Transport;
# CPAS remains a secondary, confidence-gated correction rather than the main
# source of task logits.
model['backbone']['liquid_sampler']['pair_sampler_router'][
    'relation_mode'] = 'pair_diff_product'
model['backbone']['liquid_sampler']['liquid_aware_fusion'][
    'pair_transport']['relation_mode'] = 'pair_diff_product'

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0718_06_paper_liquid_cpas_settransport_r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
