"""0702 baseline with Liquid Spectral Sampling Conv3D stem."""
from pathlib import Path

from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn import *  # noqa: F401,F403

_pairmot_root = Path(__file__).resolve().parents[4]
_pairmot_root = str(_pairmot_root)

train_dataloader['dataset'].update(
    data_root=f'{_pairmot_root}/data/hsmot/train',
)
val_dataloader['dataset'].update(
    data_root=f'{_pairmot_root}/data/hsmot/test',
)
test_dataloader = val_dataloader

load_from = str(
    f'{_pairmot_root}/pretrained_weights/'
    'o2_r18_hsmot_3dse_r2_e72_pair_dualcls_pairdn_adapted/'
    'pair_dualcls_pairdn_adapted_pretrain.pth')

_gmc_root = f'{_pairmot_root}/workdir/aux/gmc_cache'
train_dataloader['dataset'].update(
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1',
    allow_missing_gmc=False,
)
val_dataloader['dataset'].update(
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1',
    allow_missing_gmc=False,
)
test_dataloader = val_dataloader

model['backbone'].update(
    liquid_sampler=dict(
        embed_dims=32,
        tau=2.0,
        hard=False,
        init_logit=2.0,
        head_weight_std=1e-3,
        eval_hard=True,
        lowres_grad_downsample=4,
        use_lowres_grad_correction=True,
    ))

custom_hooks.append(
    dict(
        type='LiquidSamplerAnnealHook',
        tau_start=2.0,
        tau_end=0.5,
        anneal_epochs=36,
        hard_start_epoch=36,
        log_interval=200))
custom_hooks.append(dict(type='LiquidSamplerMonitorHook', interval=50))
custom_keys['backbone.stem.0.liquid_sampler'] = dict(lr_mult=1.0)
custom_keys['backbone.stem.0.se_conv'] = dict(lr_mult=1.0)
custom_keys['backbone.stem.0.se_conv1'] = dict(lr_mult=1.0)
custom_keys['backbone.stem.0.se_conv2'] = dict(lr_mult=1.0)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0703_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root=f'{_pairmot_root}/TrackEval',
    track_data_root=f'{_pairmot_root}/data/hsmot/test',
)
test_evaluator = val_evaluator
