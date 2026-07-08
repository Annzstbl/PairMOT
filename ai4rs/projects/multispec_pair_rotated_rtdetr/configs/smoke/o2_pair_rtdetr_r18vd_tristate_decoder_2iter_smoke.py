from mmengine.config import read_base

with read_base():
    from ..o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_tristate_decoder import *  # noqa: F401,F403

train_cfg = dict(type='IterBasedTrainLoop', max_iters=2, val_interval=1000)
default_hooks.checkpoint.update(interval=1000)
default_hooks.logger.update(interval=1)

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/smoke/'
    '0708_01_tristate_decoder_2iter_smoke')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
