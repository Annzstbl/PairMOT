"""Short 197 CUDA validation for the corrected hybrid AMP path."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_amp_fastgdloss_197 import *  # noqa: F401,F403

train_cfg = dict(type='IterBasedTrainLoop', max_iters=100,
                 val_interval=1000000)
val_dataloader = None
test_dataloader = None
val_cfg = None
test_cfg = None
val_evaluator = None
test_evaluator = None

train_dataloader.update(num_workers=2, persistent_workers=True)

custom_hooks = [
    hook for hook in custom_hooks
    if hook.get('type') not in (
        'EarlyStoppingHook',
        'PairTrackEarlyStoppingHook',
        'HSMOTPairValVisualizationHook',
    )
]
default_hooks.logger.update(interval=10)
default_hooks.checkpoint.update(interval=1000000, by_epoch=False)
if 'visualization' in default_hooks:
    default_hooks.visualization.update(draw=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    'tmp_validate_0714_amp_fixed_gpu5')
