"""0709 liquid unique all-GT with 8 cyclic spectral groups.

This keeps soft liquid sampling as spectral fusion, while hard/eval-hard
inference uses within-group sampling without replacement.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid import *  # noqa: F401,F403

_liquid8_patterns = [
    [7, 0, 1],
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 0],
]

model['backbone']['liquid_sampler'].update(
    num_groups=8,
    init_patterns=_liquid8_patterns,
)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0709_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
