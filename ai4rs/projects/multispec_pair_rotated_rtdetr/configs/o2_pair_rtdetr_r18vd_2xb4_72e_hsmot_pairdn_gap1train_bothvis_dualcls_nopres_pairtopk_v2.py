"""Gap-1 both-visible dual-cls/no-presence with GMC pair-topk-v2 proposals."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres import *  # noqa: F401,F403

_gmc_root = '/data/users/litianhao01/PairMmot/workdir/aux/gmc_cache'
train_dataloader['dataset'].update(
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1',
    allow_missing_gmc=False,
)
val_dataloader['dataset'].update(
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1',
    allow_missing_gmc=False,
)
test_dataloader = val_dataloader

model.update(
    query_init='pair_topk_v2',
    pair_proposal_cfg=dict(
        pre_topk=900,
        candidate_topk=1800,
        class_aware=True,
        class_mismatch_penalty=0.5,
        sim_weight=0.20,
        geom_weight=0.55,
        score_weight=0.25,
        geom_sigma=0.06,
        max_center_dist=0.18,
        max_log_scale=1.0,
        affinity_thr=0.25,
        proposal_quality_weight=0.75,
        learned_quality_weight=0.15,
        affinity_rank_weight=0.10,
    ),
)

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    '0702_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_bothvis_dualcls_nopres_pairtopk_v2')
