"""Paper Encoder with the locked class-agnostic pure-ellipse proposal rule.

This is the zero-shot c04 proposal setting evaluated on the 0727_01 epoch-72
checkpoint.  It changes no trained model weights and keeps spectral affinity
disabled.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178 import *  # noqa: F401,F403


model['pair_proposal_cfg'].update(
    elliptical_motion=True,
    ellipse_max_aspect_sqrt=1.6,
    ellipse_long_power=1.0,
    ellipse_short_power=1.0,
    ellipse_isotropic_class_ids=(),
    ellipse_isotropic_max_area=0.4e-3,
    sim_weight=0.1,
    geom_weight=0.6,
    score_weight=0.3,
    spectral_weight=0.0,
    spectral_class_ids=(),
    proposal_quality_weight=0.7,
    learned_quality_weight=0.0,
    affinity_rank_weight=0.3,
)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    'paper_encoder_pure_ellipse_c04_eval')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
