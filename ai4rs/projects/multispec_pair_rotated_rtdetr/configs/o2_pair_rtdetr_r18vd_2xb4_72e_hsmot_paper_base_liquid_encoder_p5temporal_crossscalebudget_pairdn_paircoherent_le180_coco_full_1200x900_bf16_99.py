"""0727_12: cross-scale evidence budget for Dual-Evidence encoder.

The P5 temporal MHA and all common/detail residual paths from 0727_01 are
retained. A lightweight three-token coordinator only redistributes each
branch/channel gate budget across P3/P4/P5.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_pyramidlocal_pairdn_paircoherent_le180_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


model['encoder']['post_pair_temporal_adapter_cfg'].update(
    type='pyramid_dual_evidence',
    use_spatial_evidence=False,
    conserve_branch_energy=False,
    moment_competitive_gating=False,
    cross_scale_evidence_budget=True,
    cross_scale_hidden_dims=32)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0727_12_paper_base_liquid_encoder_p5temporal_crossscalebudget_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_'
    'orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
