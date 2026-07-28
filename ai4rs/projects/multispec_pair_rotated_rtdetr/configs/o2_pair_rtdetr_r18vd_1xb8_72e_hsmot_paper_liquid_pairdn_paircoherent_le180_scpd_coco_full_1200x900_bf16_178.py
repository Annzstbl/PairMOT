"""0723_08: PairDN with spectral-coordinate pair dispersion evidence."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_independent_diffproduct_pairdn_paircoherent_le180_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


model['backbone']['liquid_sampler'][
    'spectral_coordinate_pair_dispersion'] = dict(
        max_logit_delta=0.25, eps=1e-6)
optim_wrapper['paramwise_cfg']['custom_keys'][
    'backbone.stem.0.spectral_coordinate_pair_dispersion'] = dict(
        lr_mult=1.0)

# Preserve the paper global batch of eight on the single available GPU.
train_dataloader.update(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=2)

_pairmot_root = '/data1/users/litianhao01/PairMOT'
_source_hsmot_root = '/data1/users/litianhao01/data/hsmot'
_hsmot_root = __import__('os').environ.get(
    'PAIRMOT_HSMOT_ROOT', _source_hsmot_root)
_gmc_root = f'{_pairmot_root}/workdir/aux/gmc_cache'
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

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0723_08_paper_liquid_pairdn_paircoherent_le180_scpd_r18_coco_full_'
    '1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_source_hsmot_root}/test')
test_evaluator = val_evaluator
