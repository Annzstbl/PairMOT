"""AutoDL migration of 0811_02: warmup4 plus cosine68, physical 1x8.

This keeps the validated terminal-only product-tangent model and the complete
0811_02 training protocol unchanged.  Only storage paths and the fresh workdir
are adapted to the single-GPU AutoDL instance.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0811_02_iterative_cls_terminal_transport_product_tangent_warmup4_cosine2667_decoder_178 import *  # noqa: F401,F403


_hsmot_root = '/root/autodl-tmp/data/hsmot'
_pretrain_root = '/root/autodl-fs/PairMOT_assets/pretrained_weights'
_gmc_root = '/root/autodl-tmp/PairMOT_assets/gmc_cache'
_peak_lr = 8.0e-4 / 3.0

# The source 0811_02 file omitted this assignment despite describing the same
# peak as 0810_08.  Set it explicitly here so the first warmup LR is 1e-7 and
# all existing paramwise multipliers remain ratio-preserving.
optim_wrapper['optimizer']['lr'] = _peak_lr

work_dir = (
    '/root/autodl-tmp/work_dirs/'
    '0811_02_final_product_tangent_warmup4_cosine2667_72e_1xb8_'
    'autodl_fresh_v2')
load_from = (
    f'{_pretrain_root}/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/'
    'pair_coco_adapted_pretrain.pth')
resume = False

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
