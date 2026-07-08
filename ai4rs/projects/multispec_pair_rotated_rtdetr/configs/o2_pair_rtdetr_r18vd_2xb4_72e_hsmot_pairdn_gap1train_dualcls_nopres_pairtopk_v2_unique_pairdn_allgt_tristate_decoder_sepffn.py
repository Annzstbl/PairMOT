"""0708_02: tri-state pair decoder with frame-separated FFNs.

This keeps the 0708_01 pointer/query_prev/query_curr structure but decouples
the prev/curr FFN after frame-specific cross-attention.  Proposals, matching,
DN, losses, and validation remain unchanged from the unique+allGT baseline.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_tristate_decoder import *  # noqa: F401,F403

model.decoder.update(
    tristate_separate_ffn=True,
)

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    '0708_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'tristate_decoder_sepffn')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
