"""0716_01: class-agnostic size-aware motion and spectral zero-shot."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_coco365_full_elliptical_spectral_zeroshot_99 import *  # noqa: F401,F403

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0716_01_full_sizeaware_elliptical_spectral_rank30_zeroshot')

model['pair_proposal_cfg'].update(
    elliptical_motion=True,
    ellipse_max_aspect_sqrt=1.6,
    ellipse_long_power=1.0,
    ellipse_short_power=1.0,
    ellipse_isotropic_class_ids=(),
    ellipse_isotropic_max_area=0.00035,
    sim_weight=0.10,
    geom_weight=0.65,
    score_weight=0.25,
    spectral_weight=0.04,
    spectral_sample_offset=0.20,
    spectral_affinity_mode='relative',
    spectral_relative_temperature=0.03,
    spectral_relative_positive_only=True,
    spectral_class_ids=(),
    spectral_max_pair_area=0.00035,
    spectral_pool_mode='median',
    spectral_descriptor_mode='raw_log_chroma',
    spectral_raw_weight=0.4,
    proposal_quality_weight=0.70,
    learned_quality_weight=0.0,
    affinity_rank_weight=0.30,
)

val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
