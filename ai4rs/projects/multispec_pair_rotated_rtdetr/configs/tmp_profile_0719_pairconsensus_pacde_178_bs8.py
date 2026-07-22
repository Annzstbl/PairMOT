"""Single-GPU RTX 5090 profile for the 0719_01 paper Liquid model."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_pairconsensus_relaxedset_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


_pairmot_root = '/data1/users/litianhao01/PairMOT'
_hsmot_root = '/data1/users/litianhao01/data/hsmot'
_gmc_root = f'{_pairmot_root}/workdir/aux/gmc_cache'

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    'tmp_profile_0719_pairconsensus_pacde_bs8')
load_from = (
    '/data4/litianhao/PairMmot/pretrained_weights/'
    'rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/'
    'pair_coco_adapted_pretrain.pth')
resume = False

train_dataloader.update(
    batch_size=8,
    num_workers=2,
    persistent_workers=True)
train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    ann_file=None,
    data_prefix=dict(img_path='npy2jpg'),
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1',
    allow_missing_gmc=False,
    indices=80)

train_cfg.update(max_epochs=1, val_interval=999)
val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None
default_hooks['logger'].update(interval=1)
default_hooks['checkpoint'].update(interval=999)
