"""Diagnostic zero-shot: conservative ellipse plus relative spectrum."""
from mmengine.config import read_base

with read_base():
    from .tmp_full_baseline_elliptical_motion_only_zeroshot_99 import *  # noqa: F401,F403

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    'tmp_zeroshot_full_motion_spectral_relative_conservative')

model['pair_proposal_cfg'].update(
    ellipse_max_aspect_sqrt=1.4,
    ellipse_long_power=0.75,
    ellipse_short_power=0.25,
    spectral_weight=0.03,
    spectral_affinity_mode='relative',
    spectral_relative_temperature=0.02,
    spectral_pool_mode='mean',
    spectral_descriptor_mode='raw',
)

val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
