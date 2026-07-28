"""0727_08: dual-evidence encoder with branch-energy trust regions.

The complete 0727_01 model and paper protocol are retained. Per-sample,
per-channel RMS caps only constrain common or detail updates that exceed the
corresponding input evidence.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178 import *  # noqa: F401,F403


model['encoder']['post_pair_temporal_adapter_cfg'].update(
    conserve_branch_energy=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0727_08_paper_base_liquid_encoder_p5temporal_dualbranchtrust_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
