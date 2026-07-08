"""0708_01: unique+allGT baseline with tri-state pair decoder.

The proposal generation, matching, DN, losses, and validation setup are kept
from 0704_01.  Only the pair decoder is changed from one shared query state to
three recurrent states: pointer, query_prev, and query_curr.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt import *  # noqa: F401,F403

model.decoder.update(
    tristate_decoder=True,
)

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    '0708_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'tristate_decoder')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
