"""0727_07: spatial dual-evidence with branch-energy trust regions.

This keeps the complete 0727_02 model and training protocol. Per-sample,
per-channel RMS caps prevent common and detail residuals from carrying more
energy than their respective input evidence, without adding parameters,
losses, or thresholds.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_spatialevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


model['encoder']['post_pair_temporal_adapter_cfg'].update(
    conserve_branch_energy=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0727_07_paper_base_liquid_encoder_p5temporal_spatialbranchtrust_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_'
    'orderedpairs_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
