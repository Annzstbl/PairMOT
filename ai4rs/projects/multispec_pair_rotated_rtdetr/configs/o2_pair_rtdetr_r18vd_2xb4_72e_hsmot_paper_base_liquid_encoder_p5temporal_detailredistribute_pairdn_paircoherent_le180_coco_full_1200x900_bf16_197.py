"""0727_10: detached, mean-preserving detail spatial redistribution.

The 0727_09 detail gate can change its global residual scale and backpropagate
through the evidence descriptor. This variant keeps its spatial selectivity
while restricting it to unit-mean redistribution of detached local evidence.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_detailspatial_pairdn_paircoherent_le180_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


model['encoder']['post_pair_temporal_adapter_cfg'].update(
    spatial_detach_descriptor=True,
    spatial_preserve_mean=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0727_10_paper_base_liquid_encoder_p5temporal_detailredistribute_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_'
    'orderedpairs_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
