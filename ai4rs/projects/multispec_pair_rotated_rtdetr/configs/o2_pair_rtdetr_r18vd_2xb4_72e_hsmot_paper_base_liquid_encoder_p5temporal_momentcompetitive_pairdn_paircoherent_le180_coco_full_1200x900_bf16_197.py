"""0727_11: moment-competitive dual-evidence encoder.

The 0727_01 common/detail residual paths are retained. Detached per-channel
sparsity moments augment the global descriptor, while a two-way softmax gives
common and detail evidence one shared channel budget. This targets sparse
small-object evidence and branch balance without spatial attention or losses.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_spatialevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


model['encoder']['post_pair_temporal_adapter_cfg'].update(
    use_spatial_evidence=False,
    conserve_branch_energy=False,
    moment_competitive_gating=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0727_11_paper_base_liquid_encoder_p5temporal_momentcompetitive_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_'
    'orderedpairs_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
