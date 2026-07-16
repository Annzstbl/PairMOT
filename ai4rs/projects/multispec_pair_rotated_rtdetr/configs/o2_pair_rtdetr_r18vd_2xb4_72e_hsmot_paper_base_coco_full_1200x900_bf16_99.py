"""0716_02 paper baseline: R18, COCO-only init, full HSMOT, 1200x900."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt import *  # noqa: F401,F403

_pairmot_root = '/data/users/wangying01/lth/PairMOT'
_hsmot_root = f'{_pairmot_root}/data/hsmot'
_gmc_root = f'{_pairmot_root}/workdir/aux/gmc_cache'

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0716_02_paper_base_r18_coco_full_1200x900_bf16_orderedpairs_restart')

# All trainable paper experiments use the same numerical and DDP boundary.
optim_wrapper.update(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    loss_scale=1.0)
find_unused_parameters = False
model.update(
    fp32_transformer_loss=False,
    fp32_after_encoder_loss=True)

load_from = (
    '/data4/litianhao/PairMmot/pretrained_weights/'
    'rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/'
    'pair_coco_adapted_pretrain.pth')
resume = False

# HSMOT frames are natively 1200x900. Preserve that geometry for the paper
# protocol instead of the historical effective 1067x800 resize.
for _pipeline in (train_dataloader['dataset']['pipeline'],
                  val_dataloader['dataset']['pipeline']):
    for _transform in _pipeline:
        if _transform.get('type') == 'PairSharedResize':
            _transform.update(scale=(1200, 900), keep_ratio=True)

train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    ann_file=None,
    data_prefix=dict(img_path='npy2jpg'),
    random_interval_range=None,
    frame_intervals=(1,),
    sample_seed=3407,
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1',
    allow_missing_gmc=False)
val_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/test',
    data_prefix=dict(img_path='npy2jpg'),
    frame_intervals=(1,),
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1',
    allow_missing_gmc=False)
test_dataloader = val_dataloader

# Fix the full experiment protocol, not only temporal partner sampling.
randomness = dict(seed=3407, diff_rank_seed=False, deterministic=False)
env_cfg['cudnn_benchmark'] = False
train_cfg.update(max_epochs=72, val_interval=4)
default_hooks['checkpoint'].update(interval=4, max_keep_ckpts=18)

# Paper runs must reach all 72 epochs and expose all 18 evaluation points.
custom_hooks = [
    _hook for _hook in custom_hooks
    if _hook.get('type') not in (
        'EarlyStoppingHook', 'PairTrackEarlyStoppingHook')
]

default_hooks['visualization']['draw'] = False
for _hook in custom_hooks:
    if _hook.get('type') == 'HSMOTPairValVisualizationHook':
        _hook.update(draw=False)

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
