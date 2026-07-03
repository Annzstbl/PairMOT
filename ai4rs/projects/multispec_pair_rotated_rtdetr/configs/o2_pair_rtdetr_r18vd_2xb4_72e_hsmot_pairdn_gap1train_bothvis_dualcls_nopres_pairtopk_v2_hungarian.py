"""Pair-topk-v2 trial: Hungarian one-to-one affinity proposal pairs."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2 import *  # noqa: F401,F403

model['pair_proposal_cfg'].update(
    pair_selection_mode='hungarian_affinity',
    class_aware=False,
    affinity_thr=0.15,
    proposal_quality_weight=0.85,
    learned_quality_weight=0.0,
    affinity_rank_weight=0.15,
)

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    '0702_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_bothvis_dualcls_nopres_pairtopk_v2_hungarian')
