"""Gap-1 same-index proposal ablation with baseline-like prev top-k score."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_pairtopk_sameidx_v1 import *

model.update(
    pair_proposal_cfg=dict(
        sameidx_score_mode='prev',
        sameidx_ref_source='frame',
    ),
)

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_pairtopk_sameidx_prevscore_v1')
