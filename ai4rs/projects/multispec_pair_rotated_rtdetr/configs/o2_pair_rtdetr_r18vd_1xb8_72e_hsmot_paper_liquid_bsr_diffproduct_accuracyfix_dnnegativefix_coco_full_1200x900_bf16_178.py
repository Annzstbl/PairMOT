"""0721_03 BSR-Liquid with corrected DINO negative DN sampling on 178."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_bsr_diffproduct_accuracyfix_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


# Preserve the paper global batch of eight on the single available GPU.
train_dataloader.update(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=2)

_pairmot_root = '/data1/users/litianhao01/PairMOT'
_hsmot_root = '/data1/users/litianhao01/data/hsmot'
_gmc_root = f'{_pairmot_root}/workdir/aux/gmc_cache'

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0721_03_paper_liquid_bsr_diffproduct_accuracyfix_dnnegativefix_r18_'
    'coco_full_1200x900_bf16_1xb8_fresh')

train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    ann_file=None,
    data_prefix=dict(img_path='npy2jpg'),
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1',
    allow_missing_gmc=False)
val_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/test',
    data_prefix=dict(img_path='npy2jpg'),
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1',
    allow_missing_gmc=False)
test_dataloader = val_dataloader
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
