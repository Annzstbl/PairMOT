"""AutoDL 0726_02: 0723_01 Liquid plus the 0705_01 encoder."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_pyramidlocal_pairdn_paircoherent_le180_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


_hsmot_root = '/root/autodl-tmp/data/hsmot'
_pretrain_root = '/root/autodl-fs/PairMOT_assets/pretrained_weights'
_gmc_root = '/root/autodl-tmp/PairMOT_assets/gmc_cache'

work_dir = (
    '/root/autodl-tmp/work_dirs/'
    '0726_02_paper_base_liquid_encoder_p5temporal_pyramidlocal_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_'
    'orderedpairs_autodl_1xb8_fresh')
load_from = (
    f'{_pretrain_root}/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/'
    'pair_coco_adapted_pretrain.pth')
resume = False

# One physical batch of eight preserves the original two-GPU global batch,
# optimizer, warmup, EMA, and epoch semantics without gradient accumulation.
train_dataloader.update(
    batch_size=8,
    num_workers=8,
    persistent_workers=True)
train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    ann_file=None,
    data_prefix=dict(img_path='npy2jpg'),
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1',
    allow_missing_gmc=False)
val_dataloader.update(
    batch_size=8,
    num_workers=8,
    persistent_workers=True)
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
