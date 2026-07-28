"""0727_09: Dual-Evidence with detail-only spatial reliability.

The validated common branch is unchanged. A zero-initialized, unit-output
spatial gate modulates only signed temporal detail, preserving the exact
0727_01 parent function at initialization.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_spatialevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


model['encoder']['post_pair_temporal_adapter_cfg'].update(
    use_spatial_evidence=True,
    spatial_common_evidence=False,
    spatial_detail_evidence=True,
    spatial_unit_init=True,
    conserve_branch_energy=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0727_09_paper_base_liquid_encoder_p5temporal_detailspatial_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_'
    'orderedpairs_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
