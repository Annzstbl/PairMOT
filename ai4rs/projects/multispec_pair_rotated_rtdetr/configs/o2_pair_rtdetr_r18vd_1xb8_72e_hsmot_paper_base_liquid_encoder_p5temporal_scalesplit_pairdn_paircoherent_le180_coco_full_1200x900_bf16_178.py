"""0727_03: scale-split dual-evidence temporal encoder.

The 0727_01 common branch remains active on P3/P4/P5. Signed temporal detail
is restricted to P4/P5, so high-resolution P3 improves shared detection
evidence without injecting frame-specific local differences. Liquid and the
complete paper training protocol remain unchanged.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178 import *  # noqa: F401,F403


model['encoder']['post_pair_temporal_adapter_cfg'].update(
    common_level_indices=[0, 1, 2],
    detail_level_indices=[1, 2])

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0727_03_paper_base_liquid_encoder_p5temporal_scalesplit_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
