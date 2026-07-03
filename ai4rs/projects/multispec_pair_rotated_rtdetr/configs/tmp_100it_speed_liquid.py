"""Temporary single-GPU 100-iter speed test for the liquid stem."""
from mmengine.config import read_base
from mmengine.runner.loops import IterBasedTrainLoop

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn_liquid import *  # noqa: F401,F403

train_cfg = dict(type=IterBasedTrainLoop, max_iters=100, val_interval=1000000)
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_dataloader = None
test_evaluator = None
default_hooks.logger.update(interval=10)
default_hooks.checkpoint.update(
    by_epoch=False, interval=1000000, save_last=False)
default_hooks.visualization.update(draw=False)

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/_analysis/'
    '0702_speed_100it_liquid')
