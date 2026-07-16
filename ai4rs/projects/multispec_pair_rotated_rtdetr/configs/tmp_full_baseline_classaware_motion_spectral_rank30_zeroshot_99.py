"""Diagnostic zero-shot: class-aware ellipse/spectrum with rank weight 0.30."""
from mmengine.config import read_base

with read_base():
    from .tmp_full_baseline_classaware_motion_spectral_rank25_zeroshot_99 import *  # noqa: F401,F403

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    'tmp_zeroshot_full_classaware_motion_spectral_rank30')

model['pair_proposal_cfg'].update(
    sim_weight=0.10,
    geom_weight=0.65,
    proposal_quality_weight=0.70,
    affinity_rank_weight=0.30,
)

val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
