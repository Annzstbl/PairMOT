"""0715_04 efficient pair change-gated liquid fusion on server 197.

The branch starts from wide LAF + group modulation.  Per-group spectral
coverage overlap and pooled response change gate a compact mixture of shared
and frame-specific liquid tokens.  It adds no attention or spatial pair op.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_groupmod_99 import *  # noqa: F401,F403

_pairmot_root = '/data/users/litianhao/PairMOT'
_hsmot_root = f'{_pairmot_root}/data/hsmot'
_gmc_root = f'{_pairmot_root}/workdir/aux/gmc_cache'

model['backbone']['liquid_sampler']['liquid_aware_fusion'].update(
    pair_change_gate=dict(
        hidden_dims=16,
        init_std=1e-3,
        zero_init=True,
    ))
model.update(
    fp32_transformer_loss=False,
    fp32_after_encoder_loss=True,
)

custom_keys['backbone.stem.0.liquid_aware_fusion.pair_change_gate'] = dict(
    lr_mult=1.0)

optim_wrapper.update(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    loss_scale=1.0,
)
find_unused_parameters = False
resume = False

load_from = (
    f'{_pairmot_root}/pretrained_weights/'
    'o2_r18_hsmot_3dse_r2_e72_pair_dualcls_pairdn_adapted/'
    'pair_dualcls_pairdn_adapted_pretrain.pth')

train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1',
)
val_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/test',
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1',
)
test_dataloader = val_dataloader

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0715_04_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'liquid8_laf_wide_groupmod_pairchangegate')

default_hooks['visualization']['draw'] = False
val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_hsmot_root}/test',
    trackeval_root=f'{_pairmot_root}/TrackEval',
)
test_evaluator = val_evaluator
