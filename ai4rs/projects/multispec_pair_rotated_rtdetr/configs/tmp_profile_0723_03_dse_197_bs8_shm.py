"""GPU3 concurrency profile for 197; never use for formal training."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_pairdn_paircoherent_le180_dse_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


_source_hsmot_root = '/data/users/litianhao/data/HSMOT'
_hsmot_root = __import__('os').environ.get(
    'PAIRMOT_HSMOT_ROOT', _source_hsmot_root)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    'profile_0723_03_dse_gpu3_1xb8_shm_120iter')

train_dataloader.update(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=2)
train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    ann_file=None,
    data_prefix=dict(img_path='npy2jpg'),
    indices=960)

train_cfg.update(max_epochs=1, val_interval=999)
val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None

default_hooks['logger'].update(interval=10)
default_hooks['checkpoint'].update(interval=999, save_last=False)
