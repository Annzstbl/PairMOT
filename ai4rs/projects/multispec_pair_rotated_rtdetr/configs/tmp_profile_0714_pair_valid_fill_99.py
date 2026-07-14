"""Temporary 99 smoke test for pair-valid fallback with find_unused disabled."""
from mmengine.config import read_base

with read_base():
    from .tmp_profile_0714_coco365_full_single_gpu_findunused_false import *  # noqa: F401,F403

_pairmot_root = '/data/users/wangying01/lth/PairMOT'
_hsmot_root = '/data/users/wangying01/lth/data/hsmot'
_gmc_root = f'{_pairmot_root}/workdir/aux/gmc_cache'

train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    ann_file=None,
    data_prefix=dict(img_path='npy2jpg'),
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1',
)

load_from = (
    '/data4/litianhao/PairMmot/pretrained_weights/'
    'rtdetr_r18vd_5x_coco_objects365_pair_unique_allgt_full/'
    'pair_coco365_full_adapted_pretrain.pth')

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    'tmp_profile_0714_pair_valid_fill_findunused_false')
