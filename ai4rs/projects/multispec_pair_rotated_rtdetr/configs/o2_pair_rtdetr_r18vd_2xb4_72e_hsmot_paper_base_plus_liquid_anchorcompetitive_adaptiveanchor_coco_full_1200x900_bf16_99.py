"""0718_03 ARCR Liquid with evidence-consistent anchor relaxation."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_anchorcompetitive_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


model['backbone']['liquid_sampler']['competitive_router'][
    'adaptive_anchor_relax'] = dict(
        max_relax=0.45,
        evidence_threshold=0.08,
        temperature=0.02)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0718_03_paper_base_plus_liquid_anchorcompetitive_adaptiveanchor_r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
