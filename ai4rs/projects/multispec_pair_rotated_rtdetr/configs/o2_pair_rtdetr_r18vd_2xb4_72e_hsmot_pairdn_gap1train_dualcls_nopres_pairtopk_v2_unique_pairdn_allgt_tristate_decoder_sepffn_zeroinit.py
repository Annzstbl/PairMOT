"""0708_04: zero-init tri-state decoder with frame-separated FFNs.

This combines the stable zero-initialized recurrent coupling from 0708_03
with the prev/curr FFN decoupling from 0708_02.  The goal is to keep the
tri-state decoder start less perturbed while giving the two frame queries
separate post-cross-attention capacity.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_tristate_decoder import *  # noqa: F401,F403

model.decoder.update(
    tristate_separate_ffn=True,
    tristate_zero_init_coupling=True,
)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0708_04_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'tristate_decoder_sepffn_zeroinit')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
