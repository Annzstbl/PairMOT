"""Diagnostic zero-shot: class-aware ellipse/spectrum with rank weight 0.25."""
from mmengine.config import read_base

with read_base():
    from .tmp_full_baseline_elliptical_motion_only_zeroshot_99 import *  # noqa: F401,F403

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    'tmp_zeroshot_full_classaware_motion_spectral_rank25')

model['pair_proposal_cfg'].update(
    ellipse_max_aspect_sqrt=1.6,
    ellipse_long_power=1.0,
    ellipse_short_power=1.0,
    ellipse_isotropic_class_ids=(2, ),
    sim_weight=0.15,
    geom_weight=0.60,
    score_weight=0.25,
    spectral_weight=0.04,
    spectral_affinity_mode='relative',
    spectral_relative_temperature=0.03,
    spectral_relative_positive_only=True,
    spectral_class_ids=(1, 5, 7),
    spectral_pool_mode='median',
    spectral_descriptor_mode='raw_log_chroma',
    spectral_raw_weight=0.4,
    proposal_quality_weight=0.75,
    affinity_rank_weight=0.25,
)

val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
