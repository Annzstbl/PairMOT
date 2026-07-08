"""0708_03: tri-state pair decoder with zero-init recurrent coupling.

This keeps the 0708_01 tri-state decoder but initializes the new pointer-to-
frame and frame-to-pointer coupling branches to zero.  The training start is
therefore less perturbed by randomly initialized recurrent links while the
branches can still learn through normal gradients.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_tristate_decoder import *  # noqa: F401,F403

model.decoder.update(
    tristate_zero_init_coupling=True,
)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0708_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'tristate_decoder_zeroinit')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
