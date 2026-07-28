"""0727_01: P5 temporal encoder plus dual common/detail evidence.

Liquid, PairDN, proposal, decoder, losses, and the paper data protocol are
identical to 0723_01. The 0705_01 P5 temporal branch is retained, while the
post-FPN adapter separates pair-shared detection evidence from signed temporal
detail. The two residual paths are zero-gated and frame-swap equivariant.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_pyramidlocal_pairdn_paircoherent_le180_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


model['encoder']['post_pair_temporal_adapter_cfg'].update(
    type='pyramid_dual_evidence',
    use_spatial_evidence=False)

# A physical batch of eight preserves the paper global batch on one GPU.
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
    '0727_01_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_source_hsmot_root}/test')
test_evaluator = val_evaluator
