"""0719_02 reliability-weighted pair-consensus Liquid on one RTX 5090."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_pairconsensus_relaxedset_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


# The route remains exactly shared by an ordered pair. The only new variable
# is a zero-initialized, group/slot-wise estimate of relative frame quality;
# its pairwise softmax replaces equal log-mean-exp with a symmetric weighted
# log-mixture. At initialization this is exactly the 0719_01 router.
model['backbone']['liquid_sampler']['pair_consensus_router'].update(
    reliability_weighted=True)

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
    '0719_02_paper_liquid_pairconsensus_reliability_r18_coco_full_1200x900_bf16_1xb8')
load_from = (
    '/data4/litianhao/PairMmot/pretrained_weights/'
    'rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/'
    'pair_coco_adapted_pretrain.pth')
resume = False

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

optim_wrapper['paramwise_cfg']['custom_keys'].update({
    'backbone.stem.0.liquid_sampler.pair_consensus_router.reliability': dict(
        lr_mult=1.0),
})
