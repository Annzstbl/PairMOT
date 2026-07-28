"""One-epoch CSPR-DSE candidate trial on server 197 GPU3."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_pairdn_paircoherent_le180_dse_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


# CSPR enriches route evidence; DSE inherited above enriches fusion evidence.
model['backbone']['liquid_sampler'][
    'coarse_spectral_preview_router'] = dict(
        grid_size=(24, 32), detach_shared_weight=True)

train_dataloader.update(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=2)

_source_hsmot_root = '/data/users/litianhao/data/HSMOT'
_hsmot_root = __import__('os').environ.get(
    'PAIRMOT_HSMOT_ROOT', _source_hsmot_root)
train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    ann_file=None,
    data_prefix=dict(img_path='npy2jpg'))

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    'trial_0723_cspr_dse_pairdn_gpu3_1xb8_1ep_shm')

train_cfg.update(max_epochs=1, val_interval=999)
val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None

default_hooks['logger'].update(interval=20)
default_hooks['checkpoint'].update(
    interval=1, max_keep_ckpts=1, save_last=True)
