"""Diagnostic zero-shot: elliptical motion only on full baseline."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_coco365_full_elliptical_spectral_zeroshot_99 import *  # noqa: F401,F403

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    'tmp_zeroshot_full_elliptical_motion_only')

model['pair_proposal_cfg'].update(
    elliptical_motion=True,
    sim_weight=0.20,
    geom_weight=0.55,
    score_weight=0.25,
    spectral_weight=0.0,
)

val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
