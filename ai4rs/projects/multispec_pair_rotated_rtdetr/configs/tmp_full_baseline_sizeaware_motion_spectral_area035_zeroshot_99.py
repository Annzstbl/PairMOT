"""Class-agnostic zero-shot proposal affinity with area threshold 3.5e-4."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_coco365_full_classaware_elliptical_spectral_rank30_zeroshot_99 import *  # noqa: F401,F403

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    'tmp_zeroshot_full_sizeaware_motion_spectral_area035')

model['pair_proposal_cfg'].update(
    ellipse_isotropic_class_ids=(),
    ellipse_isotropic_max_area=0.00035,
    spectral_class_ids=(),
    spectral_max_pair_area=0.00035,
)

val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
