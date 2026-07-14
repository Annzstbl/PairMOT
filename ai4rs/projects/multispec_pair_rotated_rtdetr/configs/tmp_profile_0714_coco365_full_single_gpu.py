"""Temporary single-GPU speed profile for 0714 full-data COCO365 baseline."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_coco365_full_252 import *  # noqa: F401,F403

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40, val_interval=1000000)
val_dataloader = None
test_dataloader = None
val_cfg = None
test_cfg = None
val_evaluator = None
test_evaluator = None

train_dataloader.update(batch_size=4, num_workers=2, persistent_workers=True)

model.train_cfg.assigner.update(profile_costs=True)

custom_hooks = [
    hook for hook in custom_hooks
    if hook.get('type') not in (
        'EarlyStoppingHook',
        'PairTrackEarlyStoppingHook',
        'HSMOTPairValVisualizationHook',
    )
]
custom_hooks.append(dict(type='PairComponentTimerHook', interval=5))

default_hooks.logger.update(interval=5)
default_hooks.checkpoint.update(interval=1000000, by_epoch=False)
if 'visualization' in default_hooks:
    default_hooks.visualization.update(draw=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    'tmp_profile_0714_coco365_full_single_gpu')
